
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import ModuleDict

from gpytorch.kernels import Kernel
from gpytorch.likelihoods.noise_models import Noise

from ..abstr import DGP, GPR, GPC
from ..utils import CRPS
from .svgp import SVGP



__all__ = ["SDGP", "SDGPR", "SDGPC", "DSPPR"]


class SDGPLayer(SVGP):
	"""
	sparse variational GP layer in deep Gaussian processes
	"""

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		num_induc: int,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		first_layer: bool = False,
	):
		SVGP.__init__(
			self, in_dim, out_dim, num_induc,
			True, mean_func, kernel,
		)

		self.first_layer = first_layer
	
	def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
		"""
		x ~ (S') * (B) * Q
		->
		f_mean ~ S' * B * D
		f_var ~ S' * B * D
		(S' = 1 or S)
		"""

		# x ~ (S') * (B) * Q -> (S') * B * Q
		if x.dim() == 1:
			x = x.unsqueeze(0)
		assert x.size(-1) == self.in_dim
		if x.dim() == 2:
			# x ~ B * Q
			assert self.first_layer
			batch_size = x.size(0)
		elif x.dim() == 3:
			# x ~ S' * B * Q -> S'B * Q
			assert not self.first_layer
			batch_size = x.size(1)  # B
			x = x.contiguous().view(-1, self.in_dim)
		
		# z ~ M * Q
		induc_loc = self.induc_loc
		# mu_u ~ D * M
		induc_approx_mean = self.induc_var_dist().mean
		# Sigma_u ~ D * M * M
		induc_approx_cov = self.induc_var_dist().lazy_covariance_matrix

		# f_mean & f_var ~ D * S'B -> S' * B * D
		f_mean, f_var = self.compute_f(x, induc_loc, induc_approx_mean, induc_approx_cov)
		f_mean = f_mean.view(self.out_dim, -1, batch_size).movedim(0,-1)
		f_var = f_var.view(self.out_dim, -1, batch_size).movedim(0,-1)
		return f_mean, f_var


class SDGP(DGP):
	"""
	doubly stochastic variational formulation of deep sparse Gaussian processes
	"""

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, List[int]]],
		num_induc: Union[int, List[int]],
		parametric: bool = False,
		num_sampl: int = 16,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
	):
		DGP.__init__(
			self, in_dim, out_dim, hidden_dims,
			num_induc, parametric, num_sampl,
		)

		layerdict = dict()
		for i in range(len(self.dims) - 1):
			first_layer = True if i == 0 else False
			mean_func_i = None if i == len(self.dims) - 2 else mean_func
			layerdict[f"layer_{i}"] = SDGPLayer(
				self.dims[i], self.dims[i+1], self.num_induc[i],
				mean_func_i, kernel, first_layer,
			)
		self.layers = ModuleDict(layerdict)

	@property
	def induc_kl_div(self) -> List[Tensor]:
		# list of KL(q(u^l)||p(u^l|z^{l-1})) for l in [1,L]
		return [layer.induc_kl_div for layer in self.layers.values()]
	
	def forward(self, x: Tensor, num_sampl: Optional[int] = None) -> Tuple[Tensor, Tensor]:
		"""
		x ~ (B) * D^0
		->
		f_mean ~ S' * B * D^L
		f_var ~ S' * B * D^L
		"""
		if num_sampl is None:
			num_sampl = self.num_sampl

		f = x
		for i, layer in enumerate(self.layers.values()):
			if i != 0:
				if hasattr(self, "parametric") and self.parametric:
					f = f_mean + f_var.sqrt() * self.transform_quad_node(self.quad_nodes[i-1]).unsqueeze(1)
				else:
					f = f_mean + f_var.sqrt() * torch.randn(num_sampl, *f_var.size()[1:])
			f_mean, f_var = layer(f)

		return f_mean, f_var


class SDGPR(SDGP, GPR):

	def __init__(
		self,
		sample_size: int,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, List[int]]],
		num_induc: Union[int, List[int]],
		parametric: bool = False,
		parametric_predictive: bool = False,
		num_sampl: int = 16,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		noise_covar: Optional[Noise] = None,
		noise_constraint = None,
	):
		if parametric_predictive:
			parametric = True
		SDGP.__init__(
			self, in_dim, out_dim, hidden_dims, num_induc,
			parametric, num_sampl, mean_func, kernel,
		)
		GPR.__init__(self, sample_size, parametric_predictive, noise_covar, noise_constraint)
		
	def derive_f_y(
		self,
		x: Tensor,
		y: Optional[Tensor] = None,
		**kwargs,
	) -> Any:
		"""
		x ~ (B) * D^0
		(y ~ (B) * (D^L))
		->
		f_mean ~ S' * B * D^L
		f_var ~ S' * B * D^L
		(y ~ B * D^L)
		"""

		# f_mean & f_var ~ S' * B * D^L
		f_mean, f_var = self.forward(x, **kwargs)

		if y is not None:
			# y ~ (B) * (D^L) -> B * D^L
			if y.dim() == 0 and f_mean.size(-1) == 1:
				y = y.unsqueeze(0)
			if y.dim() == 1:
				if f_mean.size(-1) == 1 and y.numel() == f_mean.size(1):
					y = y.unsqueeze(1)
				elif f_mean.size(-1) != 1 and y.numel() == f_mean.size(-1):
					y = y.unsqueeze(0)
			assert y.size() == f_mean.size()[1:]
		
		return f_mean, f_var, y

	def elbo(
		self,
		x: Tensor,
		y: Tensor,
		beta: float = 1.0,
		combine_terms: bool = True,
	) -> Any:
		
		if not self.parametric:
			num_sampl = 1
		else:
			num_sampl = None
		
		# f_mean & f_var ~ S' * B * D^L
		# y ~ B * D^L -> S' * B * D^L
		f_mean, f_var, y = self.derive_f_y(x, y, num_sampl=num_sampl)
		y = y.unsqueeze(0).expand(f_mean.size())

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

		# sum(KL(q(u^l)||p(u^l|z^{l-1}))) for l in [1,L] divided by N
		induc_kl_div = sum(d.sum() for d in self.induc_kl_div) / self.sample_size

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

		with torch.no_grad():

			if num_sampl is None:
				num_sampl = self.num_sampl
			# f_mean & f_var ~ S' * B * D^L
			# (y ~ B * D^L)
			f_mean, f_var, y = SDGPR.derive_f_y(self, x, y, num_sampl=num_sampl)

			# (weighted) average across S'
			if hasattr(self, "parametric") and self.parametric:
				pred_mean = (f_mean * self.quad_weight.view(-1,1,1)).sum(0)
			else:
				pred_mean = f_mean.mean(0)

			if y is None:
				return pred_mean
			else:
				# mll ~ B * D^L
				mll = self.mll(f_mean, f_var, y.unsqueeze(0).expand(f_mean.size()))
				alpha = mll.max(0).values
				if hasattr(self, "parametric") and self.parametric:
					mll = alpha + ((mll - alpha).exp() * self.quad_weight.view(-1,1,1)).sum(0).log()
				else:
					mll = alpha + (mll - alpha).exp().mean(0).log()
				if not return_crps:
					return pred_mean, mll
				else:
					# crps ~ B * D^L
					total_var = f_var + self.noise_covar.noise
					omega = self.quad_weight if hasattr(self, "parametric") and self.parametric else torch.ones(num_sampl)
					crps = CRPS.gaussian_mixture(
						y, f_mean.movedim(0,-1), total_var.sqrt().movedim(0,-1), omega,
					)
					return pred_mean, mll, crps


class SDGPC(SDGP, GPC):

	def __init__(
		self,
		sample_size: int,
		in_dim: int,
		hidden_dims: Optional[Union[int, List[int]]],
		num_induc: Union[int, List[int]],
		parametric: bool = False,
		num_sampl: int = 16,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		ll_num_quad: int = 16,
	):
		SDGP.__init__(
			self, in_dim, 1, hidden_dims, num_induc,
			parametric, num_sampl, mean_func, kernel,
		)
		GPC.__init__(self, sample_size, ll_num_quad)

	def elbo(
		self,
		x: Tensor,
		y: Tensor,
		beta: float = 1.0,
		combine_terms: bool = True,
	) -> Any:

		if not self.parametric:
			num_sampl = 1
		else:
			num_sampl = None
		
		# f_mean & f_var ~ S' * B * 1
		# y ~ B * 1 -> S' * B * 1
		f_mean, f_var, y = SDGPR.derive_f_y(self, x, y, num_sampl=num_sampl)
		y = y.unsqueeze(0).expand(f_mean.size())

		# ll ~ S' * B * 1 -> S' * B
		ll = self.ell(f_mean, f_var, y).squeeze(-1)
		# (weighted) average across S' & B
		if hasattr(self, "parametric") and self.parametric:
			ll = (ll.mean(-1) * self.quad_weight).sum()
		else:
			ll = ll.mean()

		# sum(KL(q(u^l)||p(u^l|z^{l-1}))) for l in [1,L] divided by N
		induc_kl_div = sum(d.sum() for d in self.induc_kl_div) / self.sample_size

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

		with torch.no_grad():

			if num_sampl is None:
				num_sampl = self.num_sampl

			# f_mean & f_var ~ S' * B * 1
			# y ~ B * 1 -> S' * B * 1
			f_mean, f_var, y = SDGPR.derive_f_y(self, x, y, num_sampl=num_sampl)

			if hasattr(self, "parametric") and self.parametric:
				pred_class = ((f_mean.squeeze(-1) * self.quad_weight.view(-1,1)).sum(0) > 0).int()
			else:
				pred_class = (f_mean.squeeze(-1).mean(0) > 0).int()
	
			if y is None:
				return pred_class
			else:
				# ll ~ S' * B * 1 -> S' * B -> B
				y = y.unsqueeze(0).expand(f_mean.size())
				ll = self.ell(f_mean, f_var, y).squeeze(-1)
				if hasattr(self, "parametric") and self.parametric:
					ll = (ll.exp() * self.quad_weight.view(-1,1)).sum(0).log()
				else:
					ll = ll.exp().mean(0).log()
				return pred_class, ll


class DSPPR(SDGPR):

	def __init1__(
		self,
		sample_size: int,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, List[int]]],
		num_induc: Union[int, List[int]],
		num_sampl: int = 16,
		mean_func = None,
		kernel: Optional[Union[Kernel, str]] = None,
		noise_covar: Optional[Noise] = None,
		noise_constraint = None,
	):
		super().__init__(
			sample_size, in_dim, out_dim, hidden_dims, num_induc, True, True,
			num_sampl, mean_func, kernel, noise_covar, noise_constraint,
		)


