
from typing import Any, Optional, Tuple, Union

import torch
from torch import Size, Tensor
from torch.distributions import kl_divergence
from torch.nn import Parameter

from linear_operator.operators import (
    CholLinearOperator,
    LinearOperator,
    TriangularLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods.noise_models import Noise
from gpytorch.variational import MeanFieldVariationalDistribution

from ..abstr import SGP, GPR, GPC
from .svgp import SVGPR, SVGPC



__all__ = ["SOGP", "SOGPR", "SOGPC"]


class SOGP(SGP):
	"""
	sparse orthogonal Gaussian processes
	"""

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		num_induc: int,
		num_induc_1: Optional[int] = None,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
	):
		SGP.__init__(
			self, in_dim, out_dim, num_induc,
			False, None, mean_func, kernel,
		)

		if num_induc_1 is None or num_induc_1 == 0:
			self.num_induc_1 = None
		else:
			self.num_induc_1 = num_induc_1  # M_1
			# z ~ M_1 * Q
			self.induc_loc_1 = Parameter(torch.randn(num_induc_1, in_dim))
			# q(u) ~ N(mu_u, Sigma_u)
			self.induc_var_dist_1 = MeanFieldVariationalDistribution(
				num_inducing_points=num_induc_1,
				batch_shape=Size([out_dim]),
			)
		
		self.num_induc_2 = num_induc  # M_2
		# o ~ M_2 * Q
		self.induc_loc_2 = Parameter(torch.randn(num_induc, in_dim))
		# q(v) ~ N(mu_v, Sigma_v)
		self.induc_var_dist_2 = MeanFieldVariationalDistribution(
			num_inducing_points=num_induc,
			batch_shape=Size([out_dim]),
		)

		self._induc_kl_div_1 = torch.zeros([])

	@property
	def induc_kl_div_1(self) -> Tensor:
		return self._induc_kl_div_1
	
	@induc_kl_div_1.setter
	def induc_kl_div_1(self, d: Tensor):
		self._induc_kl_div_1 = d

	def compute_f_1(
		self,
		x: Tensor,
		induc_loc: Tensor,
		induc_approx_mean: Tensor,
		induc_approx_cov: LinearOperator,
	) -> Tuple[Tensor, Tensor]:
		"""
		x ~ B * Q
		z ~ M_1 * Q
		mu_u ~ D * M_1
		Sigma_u ~ D * M_1 * M_1
		->
		f_mean ~ D * B
		f_var ~ D * B
		"""

		# k(z,z) ~ D * M_1 * M_1
		induc_induc_cov = self.kernel(induc_loc)
		# k(z,x) ~ D * M_1 * B
		induc_data_cov = self.kernel(induc_loc, x)

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

		# alpha = k(z,z)^-1 @ k(z,x) ~ D * M_1 * B
		L = psd_safe_cholesky(induc_induc_cov.add_jitter(self.jitter_val).to_dense())
		alpha = CholLinearOperator(TriangularLinearOperator(L)).solve(induc_data_cov.to_dense())

		# f_mean = alpha^T @ mu_u ~ D * B
		f_mean = (alpha * induc_approx_mean.unsqueeze(-1)).sum(-2)

		# f_var = alpha^T @ Sigma_u @ alpha ~ D * B
		f_var = (induc_approx_cov @ alpha * alpha).sum(-2)
		f_var = f_var + self.jitter_val

		return f_mean, f_var
	
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

		if self.num_induc_1 is None:
			f_mean_1 = f_var_1 = 0
		else:
			# z ~ M_1 * Q
			induc_loc_1 = self.induc_loc_1
			# mu_u ~ D * M_1
			induc_approx_mean_1 = self.induc_var_dist_1().mean
			# Sigma_u ~ D * M_1 * M_1
			induc_approx_cov_1 = self.induc_var_dist_1().lazy_covariance_matrix

			f_mean_1, f_var_1 = self.compute_f_1(
				x, induc_loc_1, induc_approx_mean_1, induc_approx_cov_1,
			)

		# o ~ M_2 * Q
		induc_loc_2 = self.induc_loc_2
		# mu_v ~ D * M_2
		induc_approx_mean_2 = self.induc_var_dist_2().mean
		# Sigma_v ~ D * M_2 * M_2
		induc_approx_cov_2 = self.induc_var_dist_2().lazy_covariance_matrix

		f_mean_2, f_var_2 = self.compute_f(
			x, induc_loc_2, induc_approx_mean_2, induc_approx_cov_2,
		)

		# f_mean & f_var ~ D * B -> B * D
		f_mean = (f_mean_1 + f_mean_2).mT
		f_var = (f_var_1 + f_var_2).mT
		return f_mean, f_var


class SOGPR(SOGP, GPR):
	"""
	sparse orthogonal Gaussian process regression
	"""

	def __init__(
		self,
		sample_size: int,
		in_dim: int,
		out_dim: int,
		num_induc: int,
		num_induc_1: Optional[int] = None,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		noise_covar: Optional[Noise] = None,
		noise_constraint = None,
	):
		SOGP.__init__(
			self, in_dim, out_dim, num_induc,
			num_induc_1, mean_func, kernel,
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

		# (KL(q(u)||p(u|z)) + KL(q(v)||p(v|o))) divided by N
		induc_kl_div = self.induc_kl_div_1.sum() + self.induc_kl_div.sum()
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


class SOGPC(SOGP, GPC):

	def __init__(
		self,
		sample_size: int,
		in_dim: int,
		num_induc: int,
		num_induc_1: Optional[int] = None,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		ll_num_quad: int = 16,
	):
		SOGP.__init__(
			self, in_dim, 1, num_induc,
			num_induc_1, mean_func, kernel,
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

		# (KL(q(u)||p(u|z)) + KL(q(v)||p(v|o))) divided by N
		induc_kl_div = self.induc_kl_div_1.sum() + self.induc_kl_div.sum()
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


