
import math
from typing import Optional

import torch
from torch import Size, Tensor
from torch.nn import Module

from linear_operator.operators import DenseLinearOperator

from gpytorch.kernels import Kernel

from .utils import MLP



class LinearKernel(Module):

	def __init__(self, center: bool = False):
		super().__init__()

		self.center = center

	def forward(self, x1: Tensor, x2: Tensor = None) -> Tensor:
		if x2 is None:
			x2 = x1
		if self.center:
			x1 = x1 - torch.mean(x1, dim=-1, keepdim=True)
			x2 = x2 - torch.mean(x2, dim=-1, keepdim=True)
		x1 = DenseLinearOperator(x1)
		x2 = DenseLinearOperator(x2)
		return x1 @ x2.mT


class BaseRBFKernel(Kernel):

	has_lengthscale = False

	def __init__(self, reg_factor: Optional[float] = None, **kwargs):
		super().__init__(**kwargs)

		if reg_factor is None:
			reg_factor = 1
		self.reg_factor = reg_factor
      
	def forward(self, x1, x2, diag = False, **params):
		distance = self.covar_dist(x1, x2, square_dist=True, diag=diag, **params)
		return distance.div(-2).div(self.reg_factor).exp()


class ARDSEKernel(Kernel):

	has_lengthscale = True

	def __init__(
		self,
		batch_shape: Optional[Size] = None,
		reg_factor: Optional[float] = None,
		**kwargs,
	):
		super().__init__(None, batch_shape, **kwargs)

		if reg_factor is None:
			reg_factor = 1
		self.reg_factor = reg_factor

	def forward(self, x1, x2, diag = False, **params):
		x1_ = x1.div(self.lengthscale)
		x2_ = x2.div(self.lengthscale)
		distance = self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params).div(self.reg_factor)
		return distance.div(-2).exp()


class BaseMaternKernel(Kernel):

	has_lengthscale = False

	def __init__(
		self,
		nu: float = 2.5,
		ard_num_dims: Optional[int] = None,
		batch_shape: Optional[Size] = None,
		reg_factor: Optional[float] = None,
		**kwargs,
	):
		if nu not in {0.5, 1.5, 2.5}:
			raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
		super().__init__(ard_num_dims, batch_shape, **kwargs)

		self.nu = nu
		if reg_factor is None:
			reg_factor = 1
		self.reg_factor = reg_factor

	def forward(self, x1, x2, diag=False, **params):

		distance = self.covar_dist(x1, x2, diag=diag, **params).div(math.sqrt(self.reg_factor))
		exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

		if self.nu == 0.5:
			constant_component = 1
		elif self.nu == 1.5:
			constant_component = (math.sqrt(3) * distance).add(1)
		elif self.nu == 2.5:
			constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
		return constant_component * exp_component


class MaternKernel(Kernel):

	has_lengthscale = True

	def __init__(
		self,
		nu: float = 2.5,
		ard_num_dims: Optional[int] = None,
		batch_shape: Optional[Size] = None,
		reg_factor: Optional[float] = None,
		**kwargs,
	):
		if nu not in {0.5, 1.5, 2.5}:
			raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
		super().__init__(ard_num_dims, batch_shape, **kwargs)

		self.nu = nu
		if reg_factor is None:
			reg_factor = 1
		self.reg_factor = reg_factor

	def forward(self, x1, x2, diag=False, **params):

		x1_ = x1.div(self.lengthscale)
		x2_ = x2.div(self.lengthscale)
		distance = self.covar_dist(x1_, x2_, diag=diag, **params).div(math.sqrt(self.reg_factor))
		exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

		if self.nu == 0.5:
			constant_component = 1
		elif self.nu == 1.5:
			constant_component = (math.sqrt(3) * distance).add(1)
		elif self.nu == 2.5:
			constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
		return constant_component * exp_component


class DeepKernel(Module):

	def __init__(self, kernel: Kernel, in_dim: int, depth: int = 2):
		super().__init__()

		self.kernel = kernel
		self.nn = MLP(in_dim, in_dim, depth)

	def forward(self, x1, x2=None, **kwargs):
		h1 = self.nn(x1)
		if x2 is not None:
			h2 = self.nn(x2)
		else:
			h2 = None
		return self.kernel(x1, x2, **kwargs)


