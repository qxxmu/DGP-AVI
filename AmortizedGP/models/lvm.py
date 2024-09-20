
import math
from typing import Any, Tuple

import torch
from torch import Tensor
from torch.distributions import kl_divergence
from torch.nn import Module
import torch.nn.functional as F

from linear_operator.operators import DiagLinearOperator

from gpytorch.distributions import MultivariateNormal

from ..abstr import GPR
from ..utils import MLP



__all__ = ["GPLVM"]


class Encoder(Module):

	def __init__(self, data_dim: int, latent_dim: int, depth: int = 2):
		Module.__init__(self)

		self.data_dim = data_dim  # D
		self.latent_dim = latent_dim  # Q

		self.mean_nn = MLP(data_dim, latent_dim, hidden_dims=[latent_dim]*(depth-1))
		self.std_nn = MLP(data_dim, latent_dim, hidden_dims=[latent_dim]*(depth-1))

		self.jitter_val = 1e-6
		self._kl_div = None

	@property
	def kl_div(self) -> Tensor:
		return self._kl_div
	
	@kl_div.setter
	def kl_div(self, d: Tensor):
		self._kl_div = d

	def forward(self, y: Tensor) -> Tuple[Tensor, Tensor]:
		"""
		y ~ B * D
		->
		x_mean & x_std ~ B * Q
		"""

		x_mean = self.mean_nn(y)
		x_std = F.softplus(self.std_nn(y)) + math.sqrt(self.jitter_val)

		# p(x) ~ N(0,I)
		prior_dist = MultivariateNormal(
			mean=torch.zeros(x_mean.size()),
			covariance_matrix=DiagLinearOperator(torch.ones(x_std.size())),
		)
		# q(x) ~ N(x_mean, x_cov)
		approx_dist = MultivariateNormal(
			mean=x_mean,
			covariance_matrix=DiagLinearOperator(x_std.square()),
		)
		self.kl_div = kl_divergence(approx_dist, prior_dist)

		return x_mean, x_std


class GPLVM(Module):

	def __init__(self, gpdecoder: GPR, encoder_depth: int = 2):
		Module.__init__(self)

		self.data_dim = gpdecoder.out_dim  # D
		self.latent_dim = gpdecoder.in_dim  # Q

		self.gpdecoder = gpdecoder
		self.encoder = Encoder(self.data_dim, self.latent_dim, encoder_depth)

	def encode(self, y: Tensor, return_dist: bool = False) -> Any:

		# x_mean & x_std ~ B * Q
		x_mean, x_std = self.encoder(y)

		if return_dist:
			return x_mean, x_std
		else:
			# x ~ B * Q
			x = x_mean + x_std * torch.randn(x_std.size())
			return x

	def forward(self, y: Tensor) -> Tuple[Tensor, Tensor]:
		"""
		y ~ B * D
		->
		f_mean & f_var ~ (S') * B * D
		"""
		# x ~ B * Q
		x = self.encode(y)
		# f_mean & f_var ~ (S') * B * D
		f_mean, f_var = self.gpdecoder(x)
		return f_mean, f_var

	def elbo(self, y: Tensor, combine_terms: bool = True) -> Any:

		# x ~ B * Q
		x = self.encode(y)
		latent_kl_div = self.encoder.kl_div.mean(0)

		regr_elbo = self.gpdecoder.elbo(x, y, combine_terms=combine_terms)
		if combine_terms:
			elbo = regr_elbo - latent_kl_div
			return elbo
		else:
			return (*regr_elbo, latent_kl_div)
	
	def reconstruct(self, y: Tensor) -> Any:

		with torch.no_grad():

			# x ~ B * Q
			x = self.encode(y, return_dist=True)[0]

			# f_mean & mll ~ B * D
			f_mean, mll = self.gpdecoder.pred(x, y, return_crps=False)
			return f_mean, mll


