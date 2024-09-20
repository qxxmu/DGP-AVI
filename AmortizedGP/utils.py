
import math
import numpy as np
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict, Conv1d, LeakyReLU, Linear
import torch.nn.functional as F

from linear_operator.operators import DenseLinearOperator



class ArccosineKernel(Module):

	def __init__(self, eps: float = 1e-6):
		super().__init__()

		self.eps = eps

	@property
	def outputscale(self):
		return F.softplus(self._outputscale)

	def forward(self, x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
		x1 = F.normalize(x1, dim=-1)
		if x2 is None:
			x2 = x1
		else:
			x2 = F.normalize(x2, dim=-1)
		theta = torch.acos(x1 @ x2.mT * (1 - self.eps))
		cov = 1 - theta / torch.pi
		return DenseLinearOperator(cov)


class MLP(Module):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		depth: int = None,
		hidden_dims: List[int] = None,
	):
		super().__init__()

		if depth is None and hidden_dims is None:
			depth = 2
		
		if depth is not None and hidden_dims is None:
			hidden_dims = np.linspace(
				in_dim, out_dim, num=depth, endpoint=False,
			).round().astype(int)[1:].tolist()
		elif depth is None and hidden_dims is not None:
			depth = len(hidden_dims) + 1
		elif depth is not None and hidden_dims is not None:
			assert depth == len(hidden_dims) + 1

		dims = [in_dim, *hidden_dims, out_dim]
		self.dims = dims
		self.depth = depth

		fcdict = dict()
		for i in range(depth):
			fcdict[f"fc_{i}"] = Linear(dims[i], dims[i+1])
		self.fclayers = ModuleDict(fcdict)

		self.activ = LeakyReLU(0.2)

	def forward(self, x: Tensor) -> Tensor:
		for i, layer in enumerate(self.fclayers.values()):
			x = layer(x)
			if i + 1 < self.depth:
				x = self.activ(x)
		return x
	

class LowRankAffine(Module):

	def __init__(
		self,
		in_dim: int,
		num_induc: int,
		hidden_dim: int = None,
	):
		super().__init__()

		if hidden_dim is None:
			hidden_dim = max(1, in_dim // 2)
		dims = [in_dim, hidden_dim, in_dim * num_induc]

		fcdict = dict()
		for i in range(2):
			fcdict[f"fc_{i}"] = Linear(dims[i], dims[i+1])
		self.fclayers = ModuleDict(fcdict)

	def forward(self, x: Tensor) -> Tensor:
		for layer in self.fclayers.values():
			x = layer(x)
		return x


class ConvNN(Module):

	def __init__(
		self,
		in_dim: int,
		size: int,
		out_dim: int = None,
		depth: int = None,
		hidden_dims: List[int] = None,
	):
		super().__init__()

		if out_dim is None:
			out_dim = in_dim
		
		if depth is None and hidden_dims is None:
			depth = 2
		
		if depth is not None and hidden_dims is None:
			hidden_dims = np.linspace(
				in_dim, out_dim, num=depth, endpoint=False,
			).round().astype(int)[1:].tolist()
		elif depth is None and hidden_dims is not None:
			depth = len(hidden_dims) + 1
		elif depth is not None and hidden_dims is not None:
			assert depth == len(hidden_dims) + 1

		dims = [in_dim, *hidden_dims, out_dim]
		self.dims = dims
		self.depth = depth
		self.size = size

		convdict = dict()
		for i in range(depth):
			if i == 0:
				convdict[f"conv_{i}"] = Conv1d(1, size * dims[i+1], dims[i])
			else:
				convdict[f"conv_{i}"] = Conv1d(size, size * dims[i+1], dims[i], groups=size)
		self.convlayers = ModuleDict(convdict)

		self.activ = LeakyReLU(0.2)

	def forward(self, x: Tensor) -> Tensor:

		assert x.size(-1) == self.dims[0]
		x = x.unsqueeze(-2)

		for i, conv in enumerate(self.convlayers.values()):
			x = conv(x).view(-1, self.size, self.dims[i+1])
			if i + 1 < self.depth:
				x = self.activ(x)
		return x


class CRPS:
	"""
	continuous ranked probability score
	"""

	@staticmethod
	def A(mu: Tensor, sigma: Tensor) -> Tensor:
		x = mu / sigma
		return (
			mu * torch.special.erf(x)
			+ math.sqrt(2 / math.pi) * sigma * (x.square() / -2).exp()
		)

	@staticmethod
	def gaussian_mixture(
		y: Tensor,
		mu: Tensor,
		sigma: Tensor,
		omega: Optional[Tensor] = None,
	) -> Tensor:
		"""
		y : value ~ B^0 * ... * B^K
		mu: means ~ B^0 * ... * B^K * S
		sigma: standard deviations ~ B^0 * ... * B^K * S
		(omega: mixture weights ~ (B^0 * ... * B^K) * S)
		->
		score ~ B^0 * ... * B^K
		"""

		assert mu.size() == sigma.size() and y.size() == sigma.size()[:-1]
		if omega is None:
			omega = torch.ones(mu.size(-1))
		if omega.dim() == 1:
			assert omega.size(0) == mu.size(-1)
		else:
			assert omega.size() == mu.size()
		assert (sigma >= 0).all() and (omega >= 0).all()

		omega = omega / omega.sum(-1, keepdim=True)

		part0 = (CRPS.A(y.unsqueeze(-1) - mu, sigma) * omega).sum(-1)

		mu1 = mu.unsqueeze(-1) - mu.unsqueeze(-2)
		sigma1 = (sigma.square().unsqueeze(-1) + sigma.square().unsqueeze(-2)).sqrt()
		omega1 = omega.unsqueeze(-1) * omega.unsqueeze(-2)
		part1 = (CRPS.A(mu1, sigma1) * omega1).sum([-1,-2])

		return part0 - part1 / 2
	
	@staticmethod
	def gaussian(
		y: Tensor,
		mu: Tensor,
		sigma: Tensor,
	) -> Tensor:
		"""
		y : value ~ B^0 * ... * B^K
		mu: means ~ B^0 * ... * B^K
		sigma: standard deviations ~ B^0 * ... * B^K
		->
		score ~ B^0 * ... * B^K
		"""

		assert y.size() == mu.size() == sigma.size()
		mu = mu.unsqueeze(-1)
		sigma = sigma.unsqueeze(-1)
		return CRPS.gaussian_mixture(y, mu, sigma, omega=None)


