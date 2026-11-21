"""Stacked Hourglass + AAM inference backbone from ZhenglinZhou/STAR (CVPR 2023)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .coord_conv import CoordConvTh


WFLW_EDGE_INFO: Tuple[Tuple[bool, Tuple[int, ...]], ...] = (
    (False, tuple(range(0, 33))),          # FaceContour
    (True, tuple(range(33, 42))),          # RightEyebrow
    (True, tuple(range(42, 51))),          # LeftEyebrow
    (False, (51, 52, 53, 54)),             # NoseLine
    (False, (55, 56, 57, 58, 59)),         # Nose
    (True, tuple(range(60, 68))),          # RightEye
    (True, tuple(range(68, 76))),          # LeftEye
    (True, tuple(range(76, 88))),          # OuterLip
    (True, tuple(range(88, 96))),          # InnerLip
)


@dataclass
class STARWFLWConfig:
    """Exact WFLW architecture hyperparameters from STAR conf/alignment.py."""

    width: int = 256
    height: int = 256
    nstack: int = 4
    nlevels: int = 4
    in_channel: int = 256
    increase: int = 0
    add_coord: bool = True
    use_AAM: bool = True
    decoder_type: str = "default"
    classes_num: Tuple[int, int, int] = (98, 9, 98)
    edge_info: Tuple[Tuple[bool, Tuple[int, ...]], ...] = field(default_factory=lambda: WFLW_EDGE_INFO)


class DefaultDecoder:
    """Differentiable center-of-mass heatmap decoder yielding [-1, +1] coordinates."""

    def __init__(self, weight: float = 1.0, use_weight_map: bool = False) -> None:
        self.weight = weight
        self.use_weight_map = use_weight_map

    @staticmethod
    def _make_grid(h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        yy, xx = torch.meshgrid(
            torch.arange(h).float() / (h - 1) * 2 - 1,
            torch.arange(w).float() / (w - 1) * 2 - 1,
            indexing="ij",
        )
        return yy, xx

    def get_coords_from_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        batch, npoints, h, w = heatmap.shape
        if self.use_weight_map:
            heatmap = heatmap * self.weight

        yy, xx = self._make_grid(h, w)
        yy = yy.view(1, 1, h, w).to(heatmap)
        xx = xx.view(1, 1, h, w).to(heatmap)

        heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)
        yy_coord = (yy * heatmap).sum([2, 3]) / heatmap_sum
        xx_coord = (xx * heatmap).sum([2, 3]) / heatmap_sum
        return torch.stack([xx_coord, yy_coord], dim=-1)


class Activation(nn.Module):
    def __init__(self, kind: str = "relu", channel: Optional[int] = None) -> None:
        super().__init__()
        self.kind = kind
        self.channel = channel

        if "+" in kind:
            norm_str, act_str = kind.split("+")
        else:
            norm_str, act_str = "none", kind

        if norm_str == "in":
            self.norm_fn = F.instance_norm
        elif norm_str == "bn":
            self.norm_fn = nn.BatchNorm2d(channel)
        elif norm_str == "bn_noaffine":
            self.norm_fn = nn.BatchNorm2d(channel, affine=False, track_running_stats=True)
        elif norm_str == "none":
            self.norm_fn = None
        else:
            raise ValueError(f"Unsupported norm kind: {norm_str}")

        if act_str == "relu":
            self.act_fn = F.relu
        elif act_str == "softplus":
            self.act_fn = nn.Softplus()
        elif act_str == "exp":
            self.act_fn = torch.exp
        elif act_str == "sigmoid":
            self.act_fn = torch.sigmoid
        elif act_str == "tanh":
            self.act_fn = torch.tanh
        elif act_str == "none":
            self.act_fn = None
        else:
            raise ValueError(f"Unsupported activation kind: {act_str}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_fn is not None:
            x = self.norm_fn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

    def extra_repr(self) -> str:
        return f"kind={self.kind}, channel={self.channel}"


class ConvBlock(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        bn: bool = False,
        relu: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(
            inp_dim,
            out_dim,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=True,
        )
        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(out_dim) if bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, mid_dim: Optional[int] = None) -> None:
        super().__init__()
        if mid_dim is None:
            mid_dim = out_dim // 2
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = ConvBlock(inp_dim, mid_dim, 1, relu=False)
        self.bn2 = nn.BatchNorm2d(mid_dim)
        self.conv2 = ConvBlock(mid_dim, mid_dim, 3, relu=False)
        self.bn3 = nn.BatchNorm2d(mid_dim)
        self.conv3 = ConvBlock(mid_dim, out_dim, 1, relu=False)
        self.skip_layer = ConvBlock(inp_dim, out_dim, 1, relu=False)
        self.need_skip = inp_dim != out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_layer(x) if self.need_skip else x
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.relu(self.bn3(out))
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(
        self,
        n: int,
        f: int,
        increase: int = 0,
        up_mode: str = "nearest",
        add_coord: bool = False,
        first_one: bool = False,
        x_dim: int = 64,
        y_dim: int = 64,
    ) -> None:
        super().__init__()
        nf = f + increase
        Block = ResBlock

        if add_coord:
            self.coordconv = CoordConvTh(
                x_dim=x_dim,
                y_dim=y_dim,
                with_r=True,
                with_boundary=True,
                relu=False,
                bn=False,
                in_channels=f,
                out_channels=f,
                first_one=first_one,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.coordconv = None

        self.up1 = Block(f, f)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1 = Block(f, nf)
        self.n = n
        if self.n > 1:
            self.low2 = Hourglass(n=n - 1, f=nf, increase=increase, up_mode=up_mode, add_coord=False)
        else:
            self.low2 = Block(nf, nf)
        self.low3 = Block(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode=up_mode)

    def forward(self, x: torch.Tensor, heatmap: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.coordconv is not None:
            x = self.coordconv(x, heatmap)
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class E2HTransform(nn.Module):
    def __init__(
        self,
        edge_info: Tuple[Tuple[bool, Tuple[int, ...]], ...],
        num_points: int,
        num_edges: int,
    ) -> None:
        super().__init__()
        e2h_matrix = np.zeros([num_points, num_edges])
        for edge_id, isclosed_indices in enumerate(edge_info):
            _, indices = isclosed_indices
            for point_id in indices:
                e2h_matrix[point_id, edge_id] = 1
        e2h_tensor = torch.from_numpy(e2h_matrix).float()

        self.register_buffer("weight", e2h_tensor.view(e2h_tensor.size(0), e2h_tensor.size(1), 1, 1))
        bias = ((e2h_tensor @ torch.ones(e2h_tensor.size(1)).to(e2h_tensor)) < 0.5).to(e2h_tensor)
        self.register_buffer("bias", bias)

    def forward(self, edgemaps: torch.Tensor) -> torch.Tensor:
        return F.conv2d(edgemaps, weight=self.weight, bias=self.bias)


class StackedHGNetV1(nn.Module):
    """4-stack Hourglass with CoordConv & AAM boundary fusion (STAR CVPR 2023)."""

    def __init__(
        self,
        config: STARWFLWConfig,
        classes_num: Tuple[int, int, int],
        edge_info: Tuple[Tuple[bool, Tuple[int, ...]], ...],
        nstack: int = 4,
        nlevels: int = 4,
        in_channel: int = 256,
        increase: int = 0,
        add_coord: bool = True,
        decoder_type: str = "default",
    ) -> None:
        super().__init__()
        self.cfg = config
        self.coder_type = decoder_type
        self.decoder = DefaultDecoder()
        self.nstack = nstack
        self.add_coord = add_coord
        self.num_heats = classes_num[0]

        if self.add_coord:
            convBlock: nn.Module = CoordConvTh(
                x_dim=self.cfg.width,
                y_dim=self.cfg.height,
                with_r=True,
                with_boundary=False,
                relu=True,
                bn=True,
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
            )
        else:
            convBlock = ConvBlock(3, 64, 7, 2, bn=True, relu=True)

        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        Block = ResBlock

        self.pre = nn.Sequential(
            convBlock,
            Block(64, 128),
            pool,
            Block(128, 128),
            Block(128, in_channel),
        )

        self.hgs = nn.ModuleList(
            [
                Hourglass(
                    n=nlevels,
                    f=in_channel,
                    increase=increase,
                    add_coord=self.add_coord,
                    first_one=(i == 0),
                    x_dim=int(self.cfg.width / self.nstack),
                    y_dim=int(self.cfg.height / self.nstack),
                )
                for i in range(nstack)
            ]
        )

        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    Block(in_channel, in_channel),
                    ConvBlock(in_channel, in_channel, 1, bn=True, relu=True),
                )
                for _ in range(nstack)
            ]
        )

        self.out_heatmaps = nn.ModuleList(
            [ConvBlock(in_channel, self.num_heats, 1, relu=False, bn=False) for _ in range(nstack)]
        )

        if self.cfg.use_AAM:
            self.num_edges = classes_num[1]
            self.num_points = classes_num[2]
            self.e2h_transform = E2HTransform(edge_info, self.num_points, self.num_edges)
            self.out_edgemaps = nn.ModuleList(
                [ConvBlock(in_channel, self.num_edges, 1, relu=False, bn=False) for _ in range(nstack)]
            )
            self.out_pointmaps = nn.ModuleList(
                [ConvBlock(in_channel, self.num_points, 1, relu=False, bn=False) for _ in range(nstack)]
            )
            self.merge_edgemaps = nn.ModuleList(
                [ConvBlock(self.num_edges, in_channel, 1, relu=False, bn=False) for _ in range(nstack - 1)]
            )
            self.merge_pointmaps = nn.ModuleList(
                [ConvBlock(self.num_points, in_channel, 1, relu=False, bn=False) for _ in range(nstack - 1)]
            )
            self.edgemap_act = Activation("sigmoid", self.num_edges)
            self.pointmap_act = Activation("sigmoid", self.num_points)

        self.merge_features = nn.ModuleList(
            [ConvBlock(in_channel, in_channel, 1, relu=False, bn=False) for _ in range(nstack - 1)]
        )
        self.merge_heatmaps = nn.ModuleList(
            [ConvBlock(self.num_heats, in_channel, 1, relu=False, bn=False) for _ in range(nstack - 1)]
        )

        self.heatmap_act = Activation("in+relu", self.num_heats)
        self.inference = False

    def set_inference(self, inference: bool) -> None:
        self.inference = inference

    def forward(self, x: torch.Tensor):
        x = self.pre(x)

        y, fusionmaps = [], []
        heatmaps = None
        landmarks = None
        for i in range(self.nstack):
            hg = self.hgs[i](x, heatmap=heatmaps)
            feature = self.features[i](hg)

            heatmaps0 = self.out_heatmaps[i](feature)
            heatmaps = self.heatmap_act(heatmaps0)

            if self.cfg.use_AAM:
                pointmaps0 = self.out_pointmaps[i](feature)
                pointmaps = self.pointmap_act(pointmaps0)
                edgemaps0 = self.out_edgemaps[i](feature)
                edgemaps = self.edgemap_act(edgemaps0)
                mask = self.e2h_transform(edgemaps) * pointmaps
                fusion_heatmaps = mask * heatmaps
            else:
                fusion_heatmaps = heatmaps

            landmarks = self.decoder.get_coords_from_heatmap(fusion_heatmaps)

            if i < self.nstack - 1:
                x = x + self.merge_features[i](feature) + self.merge_heatmaps[i](heatmaps)
                if self.cfg.use_AAM:
                    x = x + self.merge_pointmaps[i](pointmaps)
                    x = x + self.merge_edgemaps[i](edgemaps)

            y.append(landmarks)
            if self.cfg.use_AAM:
                y.append(pointmaps)
                y.append(edgemaps)

            fusionmaps.append(fusion_heatmaps)

        return y, fusionmaps, landmarks
