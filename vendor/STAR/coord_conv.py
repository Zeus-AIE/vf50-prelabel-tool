"""CoordConv layers adapted for inference from ZhenglinZhou/STAR (CVPR 2023)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AddCoordsTh(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, with_r: bool = False, with_boundary: bool = False) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor: torch.Tensor, heatmap: torch.Tensor | None = None) -> torch.Tensor:
        """Append normalized (x, y[, r, boundary_x, boundary_y]) coordinate channels."""
        batch_size = input_tensor.shape[0]
        device = input_tensor.device
        dtype = input_tensor.dtype

        xx_ones = torch.ones((1, self.y_dim), dtype=torch.float32, device=device).unsqueeze(-1)
        xx_range = torch.arange(self.x_dim, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(-1)

        yy_ones = torch.ones((1, self.x_dim), dtype=torch.float32, device=device).unsqueeze(1)
        yy_range = torch.arange(self.y_dim, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        xx_channel = xx_channel / (self.x_dim - 1) * 2.0 - 1.0
        yy_channel = yy_channel / (self.y_dim - 1) * 2.0 - 1.0

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).to(dtype=dtype)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).to(dtype=dtype)

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            rr = rr / torch.max(rr)
            ret = torch.cat([ret, rr], dim=1)

        if self.with_boundary and heatmap is not None:
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)
            zero_tensor = torch.zeros_like(xx_channel)
            xx_boundary = torch.where(boundary_channel > 0.05, xx_channel, zero_tensor)
            yy_boundary = torch.where(boundary_channel > 0.05, yy_channel, zero_tensor)
            ret = torch.cat([ret, xx_boundary, yy_boundary], dim=1)

        return ret


class CoordConvTh(nn.Module):
    """CoordConv layer matching ZhenglinZhou/STAR module hierarchy."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        with_r: bool,
        with_boundary: bool,
        in_channels: int,
        out_channels: int,
        first_one: bool = False,
        relu: bool = False,
        bn: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.addcoords = AddCoordsTh(
            x_dim=x_dim,
            y_dim=y_dim,
            with_r=with_r,
            with_boundary=with_boundary,
        )
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, *args, **kwargs)
        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.with_boundary = with_boundary
        self.first_one = first_one

    def forward(self, input_tensor: torch.Tensor, heatmap: torch.Tensor | None = None) -> torch.Tensor:
        assert (self.with_boundary and not self.first_one) == (heatmap is not None)
        ret = self.addcoords(input_tensor, heatmap)
        ret = self.conv(ret)
        if self.bn is not None:
            ret = self.bn(ret)
        if self.relu is not None:
            ret = self.relu(ret)
        return ret
