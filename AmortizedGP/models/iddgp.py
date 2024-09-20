
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import kl_divergence
from torch.nn import ModuleDict, Parameter
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
from .idsgp import IDSGP
from .sdgp import DGP, SDGPR, SDGPC



__all__ = ["IDDGP", "IDDGPR", "IDDGPC"]


class IDSGPLayer(SGP):

	def __init__(
		self,
		data_dim: int,
		in_dim: int,
		out_dim: int,
		num_induc_1: Optional[int],
		num_induc_2: Optional[int],
		rule: int = 0,
		linear_induc: bool = True,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		first_layer: bool = False,
		depth: int = 2,
	):
		SGP.__init__(self, in_dim, out_dim, None, False, None, mean_func, kernel)

		self.data_dim = data_dim  # D^0

		if num_induc_1 is None or num_induc_1 == 0:
			assert rule == 0
			self.num_induc_1 = None
		else:
			self.num_induc_1 = num_induc_1  # M_1

			hd_11 = min(data_dim, num_induc_1*in_dim)
			self.induc_loc_1_nn = MLP(data_dim, num_induc_1*in_dim, hidden_dims=[hd_11]*(depth-1))
			hd_12 = min(data_dim, out_dim*num_induc_1)
			self.induc_approx_mean_1_nn = MLP(data_dim, out_dim*num_induc_1, hidden_dims=[hd_12]*(depth-1))
			self.induc_approx_std_1_nn = MLP(data_dim, out_dim*num_induc_1, hidden_dims=[hd_12]*(depth-1))

		self.linear_induc = linear_induc

		if num_induc_2 is None or num_induc_2 == 0:
			assert rule == 1
			self.num_induc_2 = None
		else:
			self.num_induc_2 = num_induc_2  # M_2

			if linear_induc:
				from torch.nn import Linear
				self.induc_loc_2_nn = Linear(in_dim, num_induc_2*in_dim, bias=True)
				self.induc_loc_2_nn.weight = Parameter(torch.eye(in_dim).repeat(num_induc_2,1))
				self.induc_loc_2_nn.bias = Parameter(torch.randn(num_induc_2*in_dim))
			else:
				self.induc_loc_2_nn = MLP(in_dim, num_induc_2*in_dim, hidden_dims=[in_dim]*(depth-1))
			hd_22 = min(in_dim, out_dim*num_induc_2)
			self.induc_approx_mean_2_nn = MLP(in_dim, out_dim*num_induc_2, hidden_dims=[hd_22]*(depth-1))
			self.induc_approx_std_2_nn = MLP(in_dim, out_dim*num_induc_2, hidden_dims=[hd_22]*(depth-1))

		self.rule = rule
		self.first_layer = first_layer

		self._induc_kl_div_1 = torch.zeros([])
		self._noisy_kl_div_1 = torch.zeros([])

	@property
	def induc_kl_div_1(self) -> Tensor:
		return self._induc_kl_div_1
	
	@induc_kl_div_1.setter
	def induc_kl_div_1(self, d: Tensor):
		self._induc_kl_div_1 = d

	def induc_loc_1(self, data: Tensor) -> Tensor:
		"""
		data ~ B * D^0
		->
		z ~ B * M_1 * Q
		"""
		return self.induc_loc_1_nn(data).view(-1, self.num_induc_1, self.in_dim)

	def induc_approx_mean_cov_1(self, data: Tensor) -> Tuple[Tensor, Tensor]:
		"""
		data ~ B * D^0
		->
		mu_u ~ B * D * M_1
		Sigma_u ~ B * D * M_1 * M_1
		"""

		induc_approx_mean = self.induc_approx_mean_1_nn(data).view(-1, self.out_dim, self.num_induc_1)
		
		_induc_approx_std = self.induc_approx_std_1_nn(data).view(-1, self.out_dim, self.num_induc_1)
		induc_approx_std = F.softplus(_induc_approx_std)
		induc_approx_cov = DiagLinearOperator(induc_approx_std.square())

		return induc_approx_mean, induc_approx_cov

	def induc_loc_2(self, x: Tensor) -> Tensor:
		"""
		x ~ B * Q
		->
		o ~ B * M_2 * Q
		"""
		return self.induc_loc_2_nn(x).view(-1, self.num_induc_2, self.in_dim)

	def induc_approx_mean_cov_2(self, x: Tensor) -> Tuple[Tensor, Tensor]:
		"""
		x ~ B * Q
		->
		mu_v ~ B * D * M_2
		Sigma_v ~ B * D * M_2 * M_2
		"""

		induc_approx_mean = self.induc_approx_mean_2_nn(x).view(-1, self.out_dim, self.num_induc_2)
		
		_induc_approx_std = self.induc_approx_std_2_nn(x).view(-1, self.out_dim, self.num_induc_2)
		induc_approx_std = F.softplus(_induc_approx_std)
		induc_approx_cov = DiagLinearOperator(induc_approx_std.square())

		return induc_approx_mean, induc_approx_cov
	
	def compute_f_1(
		self,
		x: Tensor,
		induc_loc: Tensor,
		induc_approx_mean: Tensor,
		induc_approx_cov: LinearOperator,
	) -> Tuple[Tensor, Tensor]:
		"""
		x ~ S * B * Q
		z ~ B * M_1 * Q
		mu_u ~ B * D * M_1
		Sigma_u ~ B * D * M_1 * M_1
		->
		f_1_mean ~ S * B * D
		f_1_var ~ S * B * D
		"""

		# k(z,z) ~ B * D * M_1 * M_1
		induc_induc_cov = self.kernel(induc_loc.unsqueeze(1))
		# k(z,x) ~ (S) * B * D * M_1 * 1
		induc_data_cov = self.kernel(induc_loc.unsqueeze(1), x.unsqueeze(-2).unsqueeze(-2))

		# p(u|z) ~ N(m(z), k(z,z))
		induc_prior_mean = torch.zeros(induc_approx_mean.size())
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
		self.induc_kl_div_1 = induc_kl_div

		# alpha = k(z,z)^-1 @ k(z,x) ~ (S) * B * D * M * 1
		L = psd_safe_cholesky(induc_induc_cov.add_jitter(self.jitter_val).to_dense())
		alpha = CholLinearOperator(TriangularLinearOperator(L)).solve(induc_data_cov.to_dense())

		# f_mean = alpha^T @ mu_u ~ (S) * B * D
		f_mean = (alpha.squeeze(-1) * induc_approx_mean).sum(-1)

		# f_var = alpha^T @ Sigma_u @ alpha ~ (S) * B * D
		f_var = (alpha.mT @ induc_approx_cov @ alpha).squeeze(-1,-2)
		f_var = f_var + self.jitter_val

		return f_mean, f_var

	def forward(
		self,
		data: Tensor,
		x: Tensor,
		x_induc: Tensor,
	) -> Tuple[Tensor, Tensor]:
		"""
		data ~ (B) * D^0
		x_induc ~ (B * Q)
		x_mean ~ (S') * (B) * Q
		x_var ~ (S' * B * Q)
		->
		f_mean ~ S' * B * D
		f_var ~ S' * B * D
		(S' = 1 or S)
		"""

		# data ~ (B) * D^0 -> B * D^0
		if data.dim() == 1:
			data = data.unsqueeze(0)
		assert data.dim() == 2 and data.size(1) == self.data_dim

		if self.first_layer:
			# assert x_induc is None and x is None
			# x ~ 1 * B * Q
			x = data.unsqueeze(0)
			# x_induc ~ B * Q
			x_induc = data
		else:
			# x ~ S' * B * Q
			# x_induc ~ B * Q
			assert (
				x_induc is not None
				and x.dim() == 3
				and x.size(-1) == self.in_dim
				and x_induc.size() == x.size()[1:]
			)

		if self.num_induc_1 is None:
			f_1_mean = f_1_var = 0
		else:
			# z ~ B * M_1 * Q
			induc_loc_1 = self.induc_loc_1(data)
			# mu_u ~ B * D * M_1
			# Sigma_u ~ B * D * M_1 * M_1
			induc_approx_mean_1, induc_approx_cov_1 = self.induc_approx_mean_cov_1(data)

			if not hasattr(self, "rule") or self.rule == 0:
				# f_1_mean & f_1_var ~ S' * B * D
				f_1_mean, f_1_var = self.compute_f_1(
					x, induc_loc_1, induc_approx_mean_1, induc_approx_cov_1,
				)
			else:
				# f_1_mean & f_1_var ~ S' * B * D
				f_1_mean, f_1_var = IDSGP.compute_f(
					self, x, induc_loc_1, induc_approx_mean_1, induc_approx_cov_1,
				)

		if self.num_induc_2 is None:
			f_2_mean = f_2_var = 0
		else:
			# o ~ B * M_2 * Q
			induc_loc_2 = self.induc_loc_2(x_induc)
			# mu_v ~ B * D * M_2
			# Sigma_v ~ B * D * M_2 * M_2
			induc_approx_mean_2, induc_approx_cov_2 = self.induc_approx_mean_cov_2(x_induc)

			if not hasattr(self, "rule") or self.rule == 0:
				# f_2_mean & f_2_var ~ S' * B * D
				f_2_mean, f_2_var = IDSGP.compute_f(
					self, x, induc_loc_2, induc_approx_mean_2, induc_approx_cov_2,
				)
			else:
				# f_2_mean & f_2_var ~ S' * B * D
				f_2_mean, f_2_var = self.compute_f_1(
					x, induc_loc_2, induc_approx_mean_2, induc_approx_cov_2,
				)

		f_mean = f_1_mean + f_2_mean
		f_var = f_1_var + f_2_var
		return f_mean, f_var


class IDDGP(DGP):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, List[int]]],
		num_induc_1: Optional[Union[int, List[int]]],
		num_induc_2: Optional[Union[int, List[int]]],
		parametric: bool = False,
		num_sampl: int = 16,
		ar1: bool = False,
		rule: int = 0,
		linear_induc: bool = True,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		depth: int = 2,
	):
		DGP.__init__(
			self, in_dim, out_dim, hidden_dims,
			None, parametric, num_sampl,
		)

		if isinstance(num_induc_1, int) or num_induc_1 is None:
			num_induc_1 = [num_induc_1] * (len(self.dims) - 1)
		assert len(num_induc_1) == len(self.dims) - 1
		# M_1^l for l in [1,L]
		self.num_induc_1 = num_induc_1

		if isinstance(num_induc_2, int) or num_induc_2 is None:
			num_induc_2 = [num_induc_2] * (len(self.dims) - 1)
		assert len(num_induc_2) == len(self.dims) - 1
		# M_1^l for l in [1,L]
		self.num_induc_2 = num_induc_2

		self.ar1 = ar1

		assert rule in (0,1)
		self.rule = rule

		self.linear_induc = linear_induc

		# GP layers 1,...,L
		layerdict = dict()
		for i in range(len(self.dims)-1):
			first_layer = True if i == 0 else False
			mean_func_i = None if i == len(self.dims) - 2 else mean_func
			layerdict[f"layer_{i}"] = IDSGPLayer(
				in_dim, self.dims[i], self.dims[i+1], self.num_induc_1[i], self.num_induc_2[i],
				rule, linear_induc, mean_func_i, kernel, first_layer, depth,
			)
		self.layers = ModuleDict(layerdict)

	@property
	def induc_kl_div(self) -> List[Tensor]:
		# list of KL(q(u^l)||p(u^l|z^{l-1})) & KL(q(v^l)||p(v^l|o^{l-1})) for l in [1,L]
		return [[layer.induc_kl_div_1, layer.induc_kl_div] for layer in self.layers.values()]

	@property
	def noisy_kl_div(self) -> List[Tensor]:
		return [[layer.noisy_kl_div_1, layer.noisy_kl_div] for layer in self.layers.values()]

	def forward(self, x: Tensor, num_sampl: Optional[int] = None) -> Tuple[Tensor, Tensor]:
		"""
		x ~ (B) * D^0
		->
		f_mean ~ S * B * D^L
		f_var ~ S * B * D^L
		"""

		for i, layer in enumerate(self.layers.values()):
			if i == 0:
				f_induc = f = None
			else:
				if hasattr(self, "parametric") and self.parametric:
					num_sampl = self.num_sampl
					qn = self.transform_quad_node(self.quad_nodes[i-1])
					if hasattr(self, "ar1") and self.ar1:
						if i == 1:
							f_mean = f_mean.repeat(1,num_sampl,1)
							f_var = f_var.repeat(1,num_sampl,1)
						f = f_mean + f_var.sqrt() * qn.repeat(x.size(0),1)
						f_induc = f.squeeze(0)
					else:
						f_induc = (f_mean * self.quad_weight.view(-1,1,1)).sum(0)
						f = f_mean + f_var.sqrt() * qn.unsqueeze(1)
				else:
					if num_sampl is None:
						num_sampl = self.num_sampl
					if hasattr(self, "ar1") and self.ar1:
						if i == 1:
							f_mean = f_mean.repeat(1,num_sampl,1)
							f_var = f_var.repeat(1,num_sampl,1)
						f = f_mean + f_var.sqrt() * torch.randn(f_var.size())
						f_induc = f.squeeze(0)
					else:
						f_induc = f_mean.mean(0)
						f = f_mean + f_var.sqrt() * torch.randn(num_sampl, *f_var.size()[1:])

			f_mean, f_var = layer.forward(x, f, f_induc)

		if hasattr(self, "ar1") and self.ar1:
			f_mean = f_mean.view(num_sampl, x.size(0), self.out_dim)
			f_var = f_var.view(num_sampl, x.size(0), self.out_dim)

		return f_mean, f_var


class IDDGPR(IDDGP, GPR):

	def __init__(
		self,
		sample_size: int,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, List[int]]],
		num_induc_1: Optional[Union[int, List[int]]],
		num_induc_2: Optional[Union[int, List[int]]],
		parametric: bool = False,
		parametric_predictive: bool = False,
		num_sampl: int = 16,
		ar1: bool = False,
		rule: int = 0,
		linear_induc: bool = True,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		depth: int = 2,
		noise_covar: Optional[Noise] = None,
		noise_constraint = None,
	):
		if parametric_predictive:
			parametric = True
		IDDGP.__init__(
			self, in_dim, out_dim, hidden_dims, num_induc_1, num_induc_2, parametric,
			num_sampl, ar1, rule, linear_induc, mean_func, kernel, depth,
		)
		GPR.__init__(self, sample_size, parametric_predictive, noise_covar, noise_constraint)

	def elbo(
		self,
		x: Tensor,
		y: Tensor,
		beta: float = 1.0,
		combine_terms: bool = True,
	) -> Any:

		if hasattr(self, "ar1") and self.ar1 and not self.parametric:
			num_sampl = 1
		else:
			num_sampl = None

		# f_mean & f_var ~ S' * B * D^L
		# y ~ B * D^L -> S' * B * D^L
		f_mean, f_var, y = SDGPR.derive_f_y(self, x, y, num_sampl=num_sampl)
		y = y.unsqueeze(0).expand(f_mean.size())

		# ell ~ S' * B * D
		ell = self.ell(f_mean, f_var, y)
		# sum across D and (weighted) average across S' & B
		if self.parametric:
			ell = (ell.sum(-1).mean(1) * self.quad_weight).sum()
		else:
			ell = ell.sum(-1).mean()

		# ell ~ S' * B * D
		ell = self.ell(f_mean, f_var, y)
		# sum across D and (weighted) average across S' & B
		if self.parametric_predictive:
			# log-sum-exp trick
			alpha = ell.max(0).values
			ell = (alpha + ((ell - alpha).exp() * self.quad_weight.view(-1,1,1)).sum(0).log()).sum(-1).mean()
		else:
			if self.parametric:
				ell = (ell.sum(-1).mean(1) * self.quad_weight).sum()
			else:
				ell = ell.sum(-1).mean()

		# sum(KL(q(u^l)||p(u^l|z^{l-1}))+KL(q(v^l)||p(v^l|o^{l-1}))) for l in [1,L] divided by N
		induc_kl_div = sum(d.mean(0).sum() for ds in self.induc_kl_div for d in ds)
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
		num_sampl: Optional[int] = None,
	) -> Any:
		if hasattr(self, "ar1") and self.ar1 and not self.parametric:
			num_sampl = self.num_sampl
		return SDGPR.pred(self, x, y, return_crps, num_sampl)


class IDDGPC(IDDGP, GPC):

	def __init__(
		self,
		sample_size: int,
		in_dim: int,
		hidden_dims: Optional[Union[int, List[int]]],
		num_induc_1: Optional[Union[int, List[int]]],
		num_induc_2: Optional[Union[int, List[int]]],
		parametric: bool = False,
		num_sampl: int = 16,
		ar1: bool = False,
		rule: int = 0,
		linear_induc: bool = True,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		depth: int = 2,
		ll_num_quad: int = 16,
	):
		IDDGP.__init__(
			self, in_dim, 1, hidden_dims, num_induc_1, num_induc_2, parametric,
			num_sampl, ar1, rule, linear_induc, mean_func, kernel, depth,
		)
		GPC.__init__(self, sample_size, ll_num_quad)

	def elbo(
		self,
		x: Tensor,
		y: Tensor,
		beta: float = 1.0,
		combine_terms: bool = True,
	) -> Any:
		
		if hasattr(self, "ar1") and self.ar1 and not self.parametric:
			num_sampl = 1
		else:
			num_sampl = None

		# f_mean & f_var ~ S' * B * 1
		# y ~ B * 1 -> S' * B * 1
		f_mean, f_var, y = SDGPR.derive_f_y(self, x, y, num_sampl=num_sampl)
		y = y.unsqueeze(0).expand(f_mean.size())

		# ll ~ S' * B * 1 -> S' * B -> B
		ll = self.ell(f_mean, f_var, y).squeeze(-1)
		# (weighted) average across S' & B
		if self.parametric:
			ll = (ll.mean(-1) * self.quad_weight).sum()
		else:
			ll = ll.mean()

		# sum(KL(q(u^l)||p(u^l|z^{l-1}))+KL(q(v^l)||p(v^l|o^{l-1}))) for l in [1,L] divided by N
		induc_kl_div = sum(d.mean(0).sum() for ds in self.induc_kl_div for d in ds)
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
		num_sampl: Optional[int] = None,
	) -> Any:
		if hasattr(self, "ar1") and self.ar1 and not self.parametric:
			num_sampl = self.num_sampl
		return SDGPC.pred(self, x, y, num_sampl)


