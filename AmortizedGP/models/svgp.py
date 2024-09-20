
from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor

from linear_operator.operators import CholLinearOperator, TriangularLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods.noise_models import Noise

from ..abstr import SGP, GPR, GPC
from ..utils import CRPS



__all__ = ["SVGP", "SVGPR", "SVGPC"]


class SVGP(SGP):
	"""
	sparse variational Gaussian processes
	"""

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		num_induc: int,
		mean_field: bool = False,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
	):
		SGP.__init__(
			self, in_dim, out_dim, num_induc,
			True, mean_field, mean_func, kernel,
		)

	def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
		"""
		x ~ (B) * Q
		->
		f_mean ~ B * D
		f_var ~ B * D
		"""

		# x ~ (B) * Q -> B * Q
		if x.dim() == 1:
			x = x.unsqueeze(0)
		assert x.dim() == 2 and x.size(1) == self.in_dim

		# z ~ M * Q
		induc_loc = self.induc_loc
		# mu_u ~ D * M
		induc_approx_mean = self.induc_var_dist().mean
		# Sigma_u ~ D * M * M
		induc_approx_cov = self.induc_var_dist().lazy_covariance_matrix

		# f_mean & f_var ~ D * B -> B * D
		f_mean, f_var = SGP.compute_f(self, x, induc_loc, induc_approx_mean, induc_approx_cov)
		f_mean = f_mean.mT
		f_var = f_var.mT
		return f_mean, f_var

	def prior_sample(self, x: Tensor) -> Tuple[Tensor, Tensor]:

		# x ~ (B) * Q -> B * Q
		if x.dim() == 1:
			x = x.unsqueeze(0)
		assert x.dim() == 2 and x.size(1) == self.in_dim

		# z ~ M * Q
		induc_loc = self.induc_loc

		# k(z,z) ~ D * M * M
		induc_induc_cov = self.kernel(induc_loc)
		# k(z,x) ~ D * M * B
		induc_data_cov = self.kernel(induc_loc, x)
		# diag(k(x,x)) ~ D * B
		data_var = self.kernel(x, diag=True)

		# m(z) ~ D * M
		# m(x) ~ D * B
		induc_prior_mean = torch.zeros(self.out_dim, self.num_induc)
		if self.mean_func is None:
			data_mean = torch.zeros(self.out_dim, x.size(0))
		else:
			data_mean = self.mean_func(x).mT

		# p(u|z) ~ N(m(z), k(z,z))
		induc_prior_dist = MultivariateNormal(
			mean=induc_prior_mean,
			covariance_matrix=induc_induc_cov.add_jitter(self.jitter_val),
		)
		u_sample = induc_prior_dist.rsample()
		
		# alpha = k(z,z)^-1 @ k(z,x) ~ D * M * B
		L = psd_safe_cholesky(induc_induc_cov.add_jitter(self.jitter_val).to_dense())
		alpha = CholLinearOperator(TriangularLinearOperator(L)).solve(induc_data_cov.to_dense())

		f_mean = data_mean + (alpha * u_sample.unsqueeze(-1)).sum(-2)
		f_var = data_var - (induc_data_cov * alpha).sum(-2) + self.jitter_val

		f_mean = f_mean.mT
		f_var = f_var.mT
		return f_mean, f_var


class SVGPR(SVGP, GPR):
	"""
	sparse variational Gaussian process regression
	"""

	def __init__(
		self,
		sample_size: int,
		in_dim: int,
		out_dim: int,
		num_induc: int,
		mean_field: bool = False,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		noise_covar: Optional[Noise] = None,
		noise_constraint = None,
	):
		SVGP.__init__(
			self, in_dim, out_dim, num_induc,
			mean_field, mean_func, kernel,
		)
		GPR.__init__(self, sample_size, False, noise_covar, noise_constraint)
	
	def derive_f_y(self, x: Tensor, y: Optional[Tensor]) -> Any:
		"""
		x ~ (B) * Q
		(y ~ (B) * (D))
		->
		f_mean ~ B * D
		f_var ~ B * D
		(y ~ B * D)
		"""

		# f_mean & f_var ~ B * D
		f_mean, f_var = self.forward(x)

		if y is not None:
			# y ~ (B) * (D) -> B * D
			if y.dim() == 0 and f_mean.size(1) == 1:
				y = y.unsqueeze(0)
			if y.dim() == 1:
				if f_mean.size(1) == 1 and y.numel() == f_mean.size(0):
					y = y.unsqueeze(1)
				elif f_mean.size(1) != 1 and y.numel() == f_mean.size(1):
					y = y.unsqueeze(0)
			assert y.size() == f_mean.size()
		
		return f_mean, f_var, y

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
		induc_kl_div = self.induc_kl_div.sum() / self.sample_size

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

		with torch.no_grad():

			# f_mean & f_var (& y) ~ B * D
			f_mean, f_var, y = SVGPR.derive_f_y(self, x, y)

			if y is None:
				return f_mean
			else:
				# mll ~ B * D
				mll = self.mll(f_mean, f_var, y)
				if not return_crps:
					return f_mean, mll
				else:
					# crps ~ B * D
					total_var = f_var + self.noise_covar.noise
					crps = CRPS.gaussian(y, f_mean, total_var.sqrt())
					return f_mean, mll, crps


class SVGPC(SVGP, GPC):

	def __init__(
		self,
		sample_size: int,
		in_dim: int,
		num_induc: int,
		mean_field: bool = False,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		ll_num_quad: int = 16,
	):
		SVGP.__init__(
			self, in_dim, 1, num_induc,
			mean_field, mean_func, kernel,
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
		induc_kl_div = self.induc_kl_div.sum() / self.sample_size

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

		with torch.no_grad():

			# f_mean & f_var (& y) ~ B * 1
			f_mean, f_var, y = SVGPR.derive_f_y(self, x, y)
			pred_class = (f_mean.squeeze(1) > 0).int()

			if y is None:
				return pred_class
			else:
				# ll ~ B * 1 -> B
				ll = self.ell(f_mean, f_var, y).squeeze(1)
				return pred_class, ll


