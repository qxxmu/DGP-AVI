
from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import kl_divergence, Normal
import torch.nn.functional as F

from linear_operator.operators import (
	LinearOperator,
	CholLinearOperator,
	DiagLinearOperator,
	TriangularLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods.noise_models import Noise

from ..abstr import SGP, GPR, GPC
from ..utils import MLP
from .svgp import SVGPR, SVGPC



__all__ = ["IDSGP", "IDSGPR", "IDSGPC"]


class IDSGP(SGP):
	"""
	input dependent sparse Gaussian processes
	"""

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		num_induc: int,
		mean_field: bool = False,
		linear_induc: bool = True,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		depth: int = 2,
		origin_form: bool = False,
	):
		SGP.__init__(
			self, in_dim, out_dim, num_induc,
			False, None, mean_func, kernel,
		)

		self.mean_field = mean_field
		self.depth = depth

		self.orgin_form = origin_form
		if origin_form:
			linear_induc = False
		self.linear_induc = linear_induc
		if not origin_form:
			if linear_induc:
				from torch.nn import Linear
				self.induc_loc_nn = Linear(in_dim, num_induc*in_dim, bias=True)
				#self.induc_loc_nn.weight = Parameter(torch.eye(in_dim).repeat(num_induc,1))
				#self.induc_loc_nn.bias = Parameter(torch.randn(num_induc*in_dim))
			else:
				self.induc_loc_nn = MLP(in_dim, num_induc*in_dim, hidden_dims=[in_dim]*(depth-1))
			hd = min(in_dim, out_dim*num_induc)
			self.induc_approx_mean_nn = MLP(in_dim, out_dim*num_induc, hidden_dims=[hd]*(depth-1))
			if mean_field:
				self.induc_approx_std_nn = MLP(in_dim, out_dim*num_induc, hidden_dims=[hd]*(depth-1))
			else:
				L_size = num_induc*(num_induc+1)//2
				hd = min(in_dim, out_dim*L_size)
				self.induc_approx_cov_nn = MLP(in_dim, out_dim*L_size, hidden_dims=[hd]*(depth-1))
		else:
			if mean_field:
				od = in_dim*num_induc + out_dim*num_induc * 2
			else:
				od = in_dim*num_induc + out_dim*num_induc + out_dim*num_induc*(num_induc+1)//2
			hd = min(in_dim, od)
			self.amortize_nn = MLP(in_dim, od, hidden_dims=[hd]*(depth-1))

	def induc_loc(self, x: Tensor) -> Tensor:
		"""
		x ~ B * Q
		->
		z ~ B * M * Q
		"""
		return self.induc_loc_nn(x).view(-1, self.num_induc, self.in_dim)
	
	def induc_approx_mean_cov(self, x: Tensor) -> Tuple[Tensor, Tensor]:
		"""
		x ~ B * Q
		->
		mu_u ~ B * D * M
		Sigma_u ~ B * D * M * M
		"""

		induc_approx_mean = self.induc_approx_mean_nn(x)
		induc_approx_mean = induc_approx_mean.view(-1, self.out_dim, self.num_induc)
		
		if self.mean_field:
			_induc_approx_std = self.induc_approx_std_nn(x)
			_induc_approx_std = _induc_approx_std.view(-1, self.out_dim, self.num_induc)
			induc_approx_std = F.softplus(_induc_approx_std)
			induc_approx_cov = DiagLinearOperator(induc_approx_std.square())
		else:
			_induc_approx_cov = self.induc_approx_cov_nn(x)
			L_size = self.num_induc * (self.num_induc + 1) // 2
			_induc_approx_cov = _induc_approx_cov.view(-1, self.out_dim, L_size)
			idx0, idx1 = torch.tril_indices(self.num_induc, self.num_induc)
			L = torch.zeros(x.size(0), self.out_dim, self.num_induc, self.num_induc)
			L[...,idx0,idx1] = _induc_approx_cov
			induc_approx_cov = CholLinearOperator(TriangularLinearOperator(L))
		
		return induc_approx_mean, induc_approx_cov
	
	def variational_params(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

		if self.mean_field:
			induc_loc, induc_approx_mean, _induc_approx_std = self.amortize_nn(x).split([
				self.in_dim*self.num_induc,
				self.out_dim*self.num_induc,
				self.out_dim*self.num_induc,
			], dim=1)
			_induc_approx_std = _induc_approx_std.view(-1, self.out_dim, self.num_induc)
			induc_approx_cov = DiagLinearOperator(_induc_approx_std.square())
		else:
			L_size = self.num_induc * (self.num_induc + 1) // 2
			idx0, idx1 = torch.tril_indices(self.num_induc, self.num_induc)
			L = torch.zeros(x.size(0), self.out_dim, self.num_induc, self.num_induc)
			induc_loc, induc_approx_mean, _induc_approx_cov = self.amortize_nn(x).split([
				self.in_dim*self.num_induc,
				self.out_dim*self.num_induc,
				self.out_dim*L_size,
			], dim=1)
			_induc_approx_cov = _induc_approx_cov.view(-1, self.out_dim, L_size)
			L[...,idx0,idx1] = _induc_approx_cov
			induc_approx_cov = CholLinearOperator(TriangularLinearOperator(L))

		induc_loc = induc_loc.view(-1, self.num_induc, self.in_dim)
		induc_approx_mean = induc_approx_mean.view(-1, self.out_dim, self.num_induc)

		return induc_loc, induc_approx_mean, induc_approx_cov
	
	def compute_f(
		self,
		x: Tensor,
		induc_loc: Tensor,
		induc_approx_mean: Tensor,
		induc_approx_cov: LinearOperator,
	) -> Tuple[Tensor, Tensor]:
		"""
		x ~ (S) * B * Q
		z ~ B * M * Q
		mu_u ~ B * D * M
		Sigma_u ~ B * D * M * M
		->
		f_mean ~ (S) * B * D
		f_var ~ (S) * B * D
		"""

		# k(z,z) ~ B * D * M * M
		induc_induc_cov = self.kernel(induc_loc.unsqueeze(1))
		# k(z,x) ~ (S) * B * D * M * 1
		induc_data_cov = self.kernel(induc_loc.unsqueeze(1), x.unsqueeze(-2).unsqueeze(-2))
		# diag(k(x,x)) ~ (S) * B * D
		data_var = self.kernel(x.unsqueeze(-3), diag=True).mT

		# m(z) ~ B * D * M
		# m(x) ~ B * D
		if self.mean_func is None:
			induc_prior_mean = torch.zeros(induc_approx_mean.size())
			data_mean = torch.zeros(*x.size()[:-1], self.out_dim)
		else:
			induc_prior_mean = self.mean_func(induc_loc).mT
			data_mean = self.mean_func(x)

		# p(u|z) ~ N(m(z), k(z,z))
		induc_prior_dist = MultivariateNormal(
			mean=induc_prior_mean,
			covariance_matrix=induc_induc_cov.add_jitter(self.jitter_val),
		)
		# q(u) ~ N(mu_u, Sigma_u)
		induc_var_dist = MultivariateNormal(
			mean=induc_approx_mean,
			covariance_matrix=induc_approx_cov.add_jitter(self.jitter_val),
		)
		# KL(q(u)||p(u|z))
		induc_kl_div = kl_divergence(induc_var_dist, induc_prior_dist)
		self.induc_kl_div = induc_kl_div

		# alpha = k(z,z)^-1 @ k(z,x) ~ (S) * B * D * M * 1
		L = psd_safe_cholesky(induc_induc_cov.add_jitter(self.jitter_val).to_dense())
		alpha = CholLinearOperator(TriangularLinearOperator(L)).solve(induc_data_cov.to_dense())

		# f_mean = m(x) - alpha^T @ (m(z) - mu_u) ~ (S) * B * D
		mean_diff = induc_prior_mean - induc_approx_mean
		f_mean = data_mean - (alpha.squeeze(-1) * mean_diff).sum(-1)

		# f_var = Tr(k(x,x) - alpha^T @ (k(z,z) - Sigma_u) @ alpha) ~ (S) * B * D
		cov_diff = induc_induc_cov - induc_approx_cov
		f_var = data_var - (alpha.mT @ cov_diff @ alpha).squeeze(-1,-2)
		f_var = f_var + self.jitter_val

		return f_mean, f_var

	def forward(self, x: Tensor) -> Normal:
		"""
		x ~ (B) * Q
		->
		f_mean ~ D * B
		f_var ~ D * B
		"""

		# x ~ (B) * Q -> B * Q
		if x.dim() == 1:
			x = x.unsqueeze(0)
		assert x.dim() == 2 and x.size(1) == self.in_dim

		# z ~ B * M * Q
		# mu_u ~ B * D * M
		# Sigma_u ~ B * D * M * M
		if not hasattr(self, "orgin_form") or not self.orgin_form:
			induc_loc = self.induc_loc(x)
			induc_approx_mean, induc_approx_cov = self.induc_approx_mean_cov(x)
		else:
			induc_loc, induc_approx_mean, induc_approx_cov = self.variational_params(x)

		# f_mean & f_var ~ B * D
		f_mean, f_var = self.compute_f(x, induc_loc, induc_approx_mean, induc_approx_cov)
		return f_mean, f_var
	
	def prior_sample(self, x: Tensor) -> Tuple[Tensor, Tensor]:

		# x ~ (B) * Q -> B * Q
		if x.dim() == 1:
			x = x.unsqueeze(0)
		assert x.dim() == 2 and x.size(1) == self.in_dim

		# z ~ B * M * Q
		if not hasattr(self, "orgin_form") or not self.orgin_form:
			induc_loc = self.induc_loc(x)
		else:
			induc_loc = self.variational_params(x)[0]
		
		import numpy as np
		idx = np.random.choice(x.size(0), 1)
		#induc_loc = induc_loc[idx].repeat(x.size(0),1,1)
		
		# k(z,z) ~ B * D * M * M
		induc_induc_cov = self.kernel(induc_loc.unsqueeze(1))
		# k(z,x) ~ (S) * B * D * M * 1
		induc_data_cov = self.kernel(induc_loc.unsqueeze(1), x.unsqueeze(-2).unsqueeze(-2))
		# diag(k(x,x)) ~ (S) * B * D
		data_var = self.kernel(x.unsqueeze(-3), diag=True).mT

		# m(z) ~ B * D * M
		# m(x) ~ B * D
		induc_prior_mean = torch.zeros(*x.size()[:-1], self.out_dim, self.num_induc)
		if self.mean_func is None:
			data_mean = torch.zeros(*x.size()[:-1], self.out_dim)
		else:
			data_mean = self.mean_func(x)

		# p(u|z) ~ N(m(z), k(z,z))
		induc_prior_dist = MultivariateNormal(
			mean=induc_prior_mean,
			covariance_matrix=induc_induc_cov.add_jitter(self.jitter_val),
		)
		u_sample = induc_prior_dist.rsample()
		#u_sample = induc_prior_dist.rsample()[0].repeat(x.size(0),1,1)
		u_sample = self.induc_approx_mean_cov(x)[0]

		# alpha = k(z,z)^-1 @ k(z,x) ~ (S) * B * D * M * 1
		L = psd_safe_cholesky(induc_induc_cov.add_jitter(self.jitter_val).to_dense())
		alpha = CholLinearOperator(TriangularLinearOperator(L)).solve(induc_data_cov.to_dense())

		f_mean = data_mean + (alpha.squeeze(-1) * u_sample).sum(-1)
		f_var = data_var - (alpha.mT @ induc_data_cov).squeeze(-1,-2) + self.jitter_val

		return f_mean, f_var


class IDSGPR(IDSGP, GPR):

	def __init__(
		self,
		sample_size: int,
		in_dim: int,
		out_dim: int,
		num_induc: int,
		mean_field: bool = False,
		linear_induc: bool = True,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		depth: int = 2,
		origin_form: bool = False,
		noise_covar: Optional[Noise] = None,
		noise_constraint = None,
	):
		IDSGP.__init__(
			self, in_dim, out_dim, num_induc, mean_field,
			linear_induc, mean_func, kernel, depth, origin_form,
		)
		GPR.__init__(self, sample_size, False, noise_covar, noise_constraint)

	def elbo(
		self,
		x: Tensor,
		y: Tensor,
		beta: float = 1.0,
		combine_terms: bool = True,
	) -> Any:
		
		# f_mean & f_var & y ~ B * D
		f_mean, f_var, y = SVGPR.derive_f_y(self, x, y)

		# ell ~ B * D
		ell = self.ell(f_mean, f_var, y)
		# sum across D and average across B
		ell = ell.sum(1).mean()

		# KL(q(u)||p(u|z)) divided by N
		induc_kl_div = self.induc_kl_div.mean(0).sum()
		induc_kl_div = induc_kl_div / self.sample_size

		if combine_terms:
			elbo = ell - beta * induc_kl_div
			return elbo
		else:
			return ell, induc_kl_div

	def pred(
		self,
		x: Tensor,
		y: Optional[Tensor] = None,
		return_crps: bool = True,
	) -> Any:
		return SVGPR.pred(self, x, y, return_crps)


class IDSGPC(IDSGP, GPC):

	def __init__(
		self,
		sample_size: int,
		in_dim: int,
		num_induc: int,
		mean_field: bool = False,
		linear_induc: bool = True,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		depth: int = 2,
		origin_form: bool = False,
		ll_num_quad: int = 16,
	):
		IDSGP.__init__(
			self, in_dim, 1, num_induc, mean_field,
			linear_induc, mean_func, kernel, depth, origin_form,
		)
		GPC.__init__(self, sample_size, ll_num_quad)

	def elbo(
		self,
		x: Tensor,
		y: Tensor,
		beta: float = 1.0,
		combine_terms: bool = True,
	) -> Any:
		
		# f_mean & f_var & y ~ B * 1
		f_mean, f_var, y = SVGPR.derive_f_y(self, x, y)

		# ll ~ B * 1 -> B
		ll = self.ell(f_mean, f_var, y).squeeze(1)
		# average across B
		ll = ll.mean()

		# KL(q(u)||p(u|z)) divided by N
		induc_kl_div = self.induc_kl_div.mean(0).sum()
		induc_kl_div = induc_kl_div / self.sample_size

		if combine_terms:
			elbo = ll - beta * induc_kl_div
			return elbo
		else:
			return ll, induc_kl_div
	
	def pred(
		self,
		x: Tensor,
		y: Optional[Tensor] = None,
	) -> Any:
		return SVGPC.pred(self, x, y)


