
from abc import ABC, abstractmethod
import math
from numpy.polynomial.hermite_e import hermegauss
from typing import List, Optional, Tuple, Union

import torch
from torch import Size, Tensor
from torch.distributions import kl_divergence
from torch.nn import Linear, Module, Parameter, ParameterList

from linear_operator.operators import (
    CholLinearOperator,
    LinearOperator,
    TriangularLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky

from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.likelihoods.noise_models import Noise, HomoskedasticNoise
from gpytorch.variational import CholeskyVariationalDistribution, MeanFieldVariationalDistribution

from .kernels import ARDSEKernel, MaternKernel



class SGP(Module, ABC):
	"""
	abstract class of sparse Gaussian processes
	"""

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		num_induc: int,
		std_form: bool = True,
		mean_field: bool = False,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		ard: bool = False,
	):
		Module.__init__(self)

		self.in_dim = in_dim  # Q
		self.out_dim = out_dim  # D
		self.num_induc = num_induc  # M

		# m(•)
		if mean_func is None:
			self.mean_func = None
		elif mean_func == "linear":
			self.mean_func = Linear(in_dim, out_dim, bias=False)
		else:
			self.mean_func = mean_func
		
		# k(•,•)
		ard_num_dims = in_dim if ard else None
		if kernel in (None, "matern"):
			self.kernel = ScaleKernel(
				MaternKernel(batch_shape=Size([out_dim]), reg_factor=in_dim, ard_num_dims=ard_num_dims),
				batch_shape=Size([out_dim]),
			)
		elif kernel in ("rbf", "se"):
			self.kernel = ScaleKernel(
				ARDSEKernel(batch_shape=Size([out_dim]), reg_factor=in_dim, ard_num_dims=ard_num_dims),
				batch_shape=Size([out_dim]),
			)
		else:
			self.kernel = kernel

		# initialize variational parameters of standard sparse variational Gaussian processes
		if std_form:
			# z ~ M * Q
			self.induc_loc = Parameter(torch.randn(num_induc, in_dim))
			# q(u) ~ N(mu_u, Sigma_u)
			if mean_field:
				# diagonal Sigma_u
				self.induc_var_dist = MeanFieldVariationalDistribution(
					num_inducing_points=num_induc,
					batch_shape=Size([out_dim]),
				)
			else:
				# Sigma_u = L @ L^T
				self.induc_var_dist = CholeskyVariationalDistribution(
					num_inducing_points=num_induc,
					batch_shape=Size([out_dim]),
				)

		self.jitter_val = 1e-6
		self._induc_kl_div = torch.zeros([])
	
	@property
	def induc_kl_div(self) -> Tensor:
		return self._induc_kl_div
	
	@induc_kl_div.setter
	def induc_kl_div(self, d: Tensor):
		self._induc_kl_div = d

	def compute_f(
		self,
		x: Tensor,
		induc_loc: Tensor,
		induc_approx_mean: Tensor,
		induc_approx_cov: LinearOperator,
	) -> Tuple[Tensor, Tensor]:
		"""
		a common derivation for outputs of the conditional GP conditioned on inducing variables,
		i.e., a mini-batch of inputs and fixed inducing variables for every input datapoint

		x ~ B * Q
		z ~ M * Q
		mu_u ~ D * M
		Sigma_u ~ D * M * M
		->
		f_mean ~ D * B
		f_var ~ D * B
		"""

		# k(z,z) ~ D * M * M
		induc_induc_cov = self.kernel(induc_loc)
		# k(z,x) ~ D * M * B
		induc_data_cov = self.kernel(induc_loc, x)
		# diag(k(x,x)) ~ D * B
		data_var = self.kernel(x, diag=True)

		# m(z) ~ D * M
		# m(x) ~ D * B
		if self.mean_func is None:
			induc_prior_mean = torch.zeros(induc_approx_mean.size())
			data_mean = torch.zeros(self.out_dim, x.size(0))
		else:
			induc_prior_mean = self.mean_func(induc_loc).mT
			data_mean = self.mean_func(x).mT

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

		# alpha = k(z,z)^-1 @ k(z,x) ~ D * M * B
		L = psd_safe_cholesky(induc_induc_cov.add_jitter(self.jitter_val).to_dense())
		alpha = CholLinearOperator(TriangularLinearOperator(L)).solve(induc_data_cov.to_dense())

		# f_mean = m(x) - alpha^T @ (m(z) - mu_u) ~ D * B
		mean_diff = induc_prior_mean - induc_approx_mean
		f_mean = data_mean - (alpha * mean_diff.unsqueeze(-1)).sum(-2)

		# f_var = Tr(k(x,x) - alpha^T @ (k(z,z) - Sigma_u) @ alpha) ~ D * B
		cov_diff = induc_induc_cov - induc_approx_cov
		f_var = data_var - (cov_diff @ alpha * alpha).sum(-2)
		f_var = f_var + self.jitter_val

		return f_mean, f_var
	
	@abstractmethod
	def forward(self):
		pass


class DGP(SGP, ABC):
	"""
	abstract class of deep Gaussian processes
	"""

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, List[int]]],
		num_induc: Union[int, List[int]],
		parametric: bool = False,
		num_sampl: int = 16,
	):
		Module.__init__(self)

		self.in_dim = in_dim  # D^0
		self.out_dim = out_dim  # D^L
		if hidden_dims is None:
			hidden_dims = []
		elif isinstance(hidden_dims, int):
			hidden_dims = [hidden_dims]
		# [D^0,...,D^L]
		self.dims = [in_dim] + hidden_dims + [out_dim]

		if isinstance(num_induc, int) or num_induc is None:
			num_induc = [num_induc] * (len(self.dims) - 1)
		assert len(num_induc) == len(self.dims) - 1
		# M^l for l in [1,L]
		self.num_induc = num_induc

		self.num_sampl = num_sampl  # S

		self.parametric = parametric
		if parametric:
			quad_node_dict = []
			# xi^l ~ S * D^l for l in [1,L-1]
			for i in range(len(self.dims)-2):
				quad_node_dict.append(Parameter(torch.randn(num_sampl, self.dims[i+1])))
			self.quad_nodes = ParameterList(quad_node_dict)
			self.max_xi = 5.
			# raw_omega ~ S-1
			self._quad_weight = Parameter(torch.zeros(num_sampl-1))

	def transform_quad_node(self, qn: Tensor) -> Tensor:
		if hasattr(self, "max_xi") and self.max_xi is not None:
			abs_max = abs(self.max_xi)
		else:
			abs_max = 5.
		return (torch.special.expit(qn) * 2 - 1) * abs_max

	@property
	def quad_weight(self) -> Tensor:
		# non-negative omega ~ S, sum(omega) = 1
		qwt = torch.cat([self._quad_weight.exp(), torch.ones(1)])
		qwt = qwt / qwt.sum()
		return qwt

	@property
	@abstractmethod
	def induc_kl_div(self):
		pass

	@abstractmethod
	def forward(self):
		pass


class GPR(ABC):
	"""
	standard & parametric predictive Gaussian process regressions
	"""

	def __init__(
		self,
		sample_size: int,
		parametric_predictive: bool = False,
		noise_covar: Optional[Noise] = None,
		noise_constraint = None,
	):
		assert issubclass(self.__class__, SGP)

		self.sample_size = sample_size  # N

		# whether to adopt parametric predictive Gaussian process regression
		self.parametric_predictive = parametric_predictive

		# sigma_obs^2
		if noise_constraint is None:
			noise_constraint = GreaterThan(1e-4)
		if noise_covar is None:
			noise_covar = HomoskedasticNoise(noise_constraint=noise_constraint)
		self.add_module("noise_covar", noise_covar)
	
	def mll(self, f_mean: Tensor, f_var: Tensor, y: Tensor) -> Tensor:
		"""
		marginal log-likelihood of data
		"""

		assert f_mean.size() == f_var.size() == y.size()

		total_var = f_var + self.noise_covar.noise
		log_prob = - (math.log(2 * math.pi) + total_var.log()
			+ (y - f_mean).square() / total_var) / 2
		
		return log_prob

	def ell(self, f_mean: Tensor, f_var: Tensor, y: Tensor) -> Tensor:
		"""
		expected log-likelihood term in training objective
		"""

		assert f_mean.size() == f_var.size() == y.size()

		if self.parametric_predictive:
			# parametric predictive GP regression
			log_prob = self.mll(f_mean, f_var, y)
		else:
			# standard GP regression
			inv_noise = self.noise_covar.noise.reciprocal()
			log_prob = (inv_noise.log() - math.log(2 * math.pi)
				- ((y - f_mean).square() + f_var) * inv_noise) / 2
		
		return log_prob

	@abstractmethod
	def elbo(self):
		pass

	@abstractmethod
	def pred(self):
		pass


class GPC(ABC):

	def __init__(
		self,
		sample_size: int,
		ll_num_quad: int = 16,
	):
		assert issubclass(self.__class__, SGP)

		self.sample_size = sample_size  # N
		self.num_class = self.out_dim + 1  # C = D - 1
		# Hermite (probabilist's)
		self.init_quad(ll_num_quad)

		self.shrinkage = 1e-3

	def init_quad(self, num_quad: int):
		self.ll_num_quad = num_quad  # S_ll
		# xi & omega ~ S_ll
		quad_node, quad_weight = hermegauss(num_quad)
		self.ll_quad_node = torch.tensor(quad_node)
		self.ll_quad_weight = torch.tensor(quad_weight)

	def ell(self, f_mean: Tensor, f_var: Tensor, y: Tensor) -> Tensor:

		assert (
			f_mean.size() == f_var.size() == y.size()
			and y.size(-1) == 1
		)

		f_quad = f_mean + f_var.sqrt() * self.ll_quad_node
		phi = torch.special.erf(f_quad / math.sqrt(2)) / 2 + 0.5
		phi = (phi - 0.5) * (1 - self.shrinkage) + 0.5
		log_prob = y * phi.log() + (1 - y) * (1 - phi).log()
		log_prob = (log_prob * self.ll_quad_weight).sum(-1, keepdim=True) / math.sqrt(2 * math.pi)

		return log_prob

	@abstractmethod
	def elbo(self):
		pass

	@abstractmethod
	def pred(self):
		pass


