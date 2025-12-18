# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "C3k2_MLCA",
    "C2TSSA_DYT_Mona_EDFFN",
    "ELA_HSFPN",
    "Multiply",
    "Add",
    "Fusion",
    "C2SFA",
    "C2PTSSA"
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))
class DOAA(nn.Module):
    """密集目标感知注意力 (Dense Object-Aware Attention)
    
    专门设计用于提升密集目标检测性能的注意力机制，结合空间和通道注意力，
    并增加密度感知和局部对比度增强。
    """
    def __init__(self, c, r=16):
        super().__init__()
        # 通道压缩率
        self.channels = c
        
        # 空间密度感知分支
        self.density_branch = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c),
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.Conv2d(c, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道注意力分支
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(c // r, c, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 局部对比度增强 - 使用深度可分离卷积提取局部特征差异
        self.contrast_conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2, groups=c),
            nn.Conv2d(c, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )
        
        # 最终融合
        self.fusion = nn.Conv2d(c, c, kernel_size=1)
        
    def forward(self, x):
        # 密度感知
        density_map = self.density_branch(x)
        
        # 通道注意力
        channel_att = self.channel_gate(x)
        
        # 局部对比度增强
        contrast_feat = self.contrast_conv(x)
        
        # 自适应融合：在密集区域增强对比度特征
        enhanced = x * channel_att + contrast_feat * density_map
        
        return self.fusion(enhanced)


class Bottleneck_DOAA(Bottleneck):
    """带密集目标感知注意力的Bottleneck"""
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.attention = DOAA(c2)
        
    def forward(self, x):
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C2f_DOAA(C2f):
    """带密集目标感知注意力的C2f模块"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            Bottleneck_DOAA(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )
        
    def forward(self, x):
        # 保持与C2f相同的前向过程，便于加载预训练权重
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# 新增：C3_DOAA - 继承 C3，并集成 DOAA 通过 Bottleneck_DOAA
class C3_DOAA(C3):
    """带密集目标感知注意力的C3模块 (CSP Bottleneck with 3 convolutions)."""
    
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions and DOAA.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        # 修改：使用 Bottleneck_DOAA 替换原 Bottleneck
        self.m = nn.Sequential(*(Bottleneck_DOAA(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions and DOAA."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# 新增：C3k_DOAA - 继承 C3_DOAA，并支持自定义 kernel size (k)
class C3k_DOAA(C3_DOAA):
    """带密集目标感知注意力的C3k模块 (C3 with customizable kernel sizes)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """
        Initialize C3k_DOAA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # 修改：使用 Bottleneck_DOAA，并调整 kernel size 为 (k, k)
        self.m = nn.Sequential(*(Bottleneck_DOAA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# 新增：C3k2_DOAA - 继承 C3k2，并集成 DOAA 通过 C3k_DOAA 或 Bottleneck_DOAA
class C3k2_DOAA(C3k2):
    """带密集目标感知注意力的C3k2模块 (Faster Implementation of CSP Bottleneck with 2 convolutions)."""
    
    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2_DOAA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k_DOAA blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        # 修改：使用 C3k_DOAA (if c3k) 或 Bottleneck_DOAA 替换原 self.m
        self.m = nn.ModuleList(
            C3k_DOAA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DOAA(self.c, self.c, shortcut, g) for _ in range(n)
        )


# CFC_CRB=CFC_CRB_Enhanced
# SFC_G2=SFC_G2_Enhanced

class GAMAttention(nn.Module):

    def __init__(self, c1, c2, group=True, rate=4):
        super(GAMAttention, self).__init__()
        
        # 确保参数有效性
        self.c1 = c1
        self.c2 = c2
        
        # 修复：当c1较小时，自适应调整rate以避免隐藏通道过小
        if c1 <= 32:
            self.rate = max(1, c1 // 8)  # 确保至少有8个隐藏通道
        else:
            self.rate = rate
        self.rate = max(1, self.rate)  # 确保rate至少为1
        
        # 计算隐藏通道数，确保至少为8（对于小模型更稳定）
        self.hidden_channels = max(8, int(c1 / self.rate))
        
        # 通道注意力 - 修复维度计算
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, c1),
        )
        
        # 空间注意力 - 根据通道数智能选择组卷积
        if group and c1 >= 8 and c1 % 4 == 0:
            # 只有当通道数足够且能被4整除时才使用组卷积
            groups1 = min(4, c1 // 4)
            groups2 = min(4, self.hidden_channels // 2) if self.hidden_channels >= 8 else 1
            self.spatial_conv1 = nn.Conv2d(c1, self.hidden_channels, kernel_size=7, padding=3, groups=groups1)
            self.spatial_conv2 = nn.Conv2d(self.hidden_channels, c1, kernel_size=7, padding=3, groups=groups2)
        else:
            # 使用标准卷积，对所有情况都适用
            self.spatial_conv1 = nn.Conv2d(c1, self.hidden_channels, kernel_size=7, padding=3)
            self.spatial_conv2 = nn.Conv2d(self.hidden_channels, c1, kernel_size=7, padding=3)
            
        self.spatial_attention = nn.Sequential(
            self.spatial_conv1,
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            self.spatial_conv2,
            nn.BatchNorm2d(c1),
        )
        
        # 通道调整层（如果输入输出通道不同）
        self.channel_adjust = None
        if c1 != c2:
            self.channel_adjust = Conv(c1, c2, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 验证输入维度 - 提供更详细的错误信息
        if c != self.c1:
            print(f"GAMAttention维度不匹配: 期望{self.c1}, 实际{c}, rate={self.rate}, hidden={self.hidden_channels}")
            raise ValueError(f"输入通道数 {c} 与期望的 {self.c1} 不匹配")
        
        # 通道注意力 - 修复矩阵维度问题
        # 将空间维度展平，但保持批次和通道维度分离
        x_permute = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_reshape = x_permute.view(b * h * w, c)  # [B*H*W, C]
        x_att = self.channel_attention(x_reshape)  # [B*H*W, C]
        x_att_permute = x_att.view(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]
        x_channel_att = x_att_permute.sigmoid()
        x = x * x_channel_att

        # 空间注意力
        x_spatial_att = self.spatial_attention(x).sigmoid()
        # 应用通道混洗（如果通道数足够）
        if c >= 4:
            x_spatial_att = channel_shuffle(x_spatial_att, min(4, c))
        out = x * x_spatial_att
        
        # 通道调整（如果需要）
        if self.channel_adjust is not None:
            out = self.channel_adjust(out)
        
        return out


def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out

class C2f_GAM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # 修复：GAMAttention应该应用在cv2的输出上，使用c2作为输入通道
        self.att = GAMAttention(c2, c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))
class C3k2_GAM(C2f_GAM):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
C3K2_MOD = C3k2_GAM

######################################
class ELA_HSFPN(nn.Module):
    def __init__(self, in_planes, flag=True):
        super(ELA_HSFPN, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(in_planes, in_planes, 7, padding=3),
            nn.GroupNorm(16, in_planes),
            nn.Sigmoid()
        )
        self.flag = flag
    
    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv1x1(self.pool_h(x).reshape((b, c, h))).reshape((b, c, h, 1))
        x_w = self.conv1x1(self.pool_w(x).reshape((b, c, w))).reshape((b, c, 1, w))
        return x * x_h * x_w if self.flag else x_h * x_w

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class Multiply(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x[0] * x[1]
class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(torch.stack(x, dim=0), dim=0)
    


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

class SDI(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
        self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[2:]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans

class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()
        
        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI']
        self.fusion = fusion
        
        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == 'SDI':
            self.SDI = SDI(inc_list)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
        
    
    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'SDI':
            return self.SDI(x)
######################################
import math
class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma = 2, b = 1,local_weight=0.5):
        super(MLCA, self).__init__()

        # ECA 计算方法
        self.local_size=local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)   # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight=local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv=self.local_arv_pool(x)
        global_arv=self.global_arv_pool(local_arv)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)


        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)

        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(),[self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [m, n])

        x=x * att_all
        return x

class Bottleneck_MLCA(Bottleneck):
    """Standard bottleneck with FocusedLinearAttention."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        self.attention = MLCA(c2)
    
    def forward(self, x):
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))

class C3k_MLCA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_MLCA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_MLCA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MLCA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_MLCA(self.c, self.c, shortcut, g) for _ in range(n))



###################################
from einops import rearrange, repeat

class PSABlock_EDFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = EDFFN(c, 2, False)

class C2PSA_EDFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_EDFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

class AttentionTSSA(nn.Module):
    # https://github.com/RobinWu218/ToST
    def __init__(self, dim, num_heads = 8, qkv_bias=False, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        
        self.heads = num_heads

        self.attend = nn.Softmax(dim = 1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )
    
    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h = self.heads)

        b, h, N, d = w.shape
        
        w_normed = torch.nn.functional.normalize(w, dim=-2) 
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp) # b * h * n 
        
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
from .mona import Mona,MonaOp
class TSSAlock_DYT_Mona_EDFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = EDFFN(c, ffn_expansion_factor=2, bias=False)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = self.mona1(x)
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        x = self.mona2(x)
        return x

class C2TSSA_DYT_Mona_EDFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(TSSAlock_DYT_Mona_EDFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


################################
class LayerNorm2d(nn.LayerNorm):
    """
    Channel-First Layer Normalization for 2D data (B, C, H, W).
    """
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

from timm.models.layers import DropPath, trunc_normal_
class TSSAlock_ParallelFFN(PSABlock):
    """
    改进方案一：
    将 EDFFN 和 Mona 视为并行的 FFN 专家。
    TSSA (Attention) -> [ EDFFN | Mona ] (Parallel FFNs)
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, drop_path=0.1) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        # 1. TSSA 注意力模块 (来自你的代码)
        self.attn = AttentionTSSA(c, num_heads=num_heads)
        
        # 2. 并行的 FFN 专家
        self.ffn = EDFFN(c, ffn_expansion_factor=2, bias=False)  # 频域专家
        self.mona = Mona(c)                                      # 多尺度空间专家

        # 3. 标准组件
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        BS, C, H, W = x.size()
        
        # 1. TSSA 注意力分支 (标准 Transformer 块的前半部分)
        attn_in = self.norm1(x)
        attn_out = self.attn(attn_in.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.drop_path(attn_out) if self.add else self.drop_path(attn_out)

        # 2. 并行 FFN 专家分支 (标准 Transformer 块的后半部分)
        norm_x = self.norm2(x)
        
        # 两个专家独立工作
        ffn_out = self.ffn(norm_x)
        mona_out = self.mona(norm_x)
        
        # 融合专家结果
        x = x + self.drop_path(ffn_out) + self.drop_path(mona_out) if self.add else self.drop_path(ffn_out) + self.drop_path(mona_out)
        
        return x

class C2TSSA_ParallelFFN(C2PSA):
    """
    使用 TSSAlock_ParallelFFN 的 C2 模块
    """
    def __init__(self, c1, c2, n=1, e=0.5, drop_path=0.1):
        super().__init__(c1, c2, n, e)
        
        # 构建 n 个并行的 Block
        self.m = nn.Sequential(*(TSSAlock_ParallelFFN(self.c, 
                                                       attn_ratio=0.5, 
                                                       num_heads=max(1, self.c // 64), 
                                                       drop_path=drop_path) 
                                 for _ in range(n)))
###########################################################



class HierarchicalMona(nn.Module):

    """修复版：简化冗余逻辑"""

    def __init__(self, in_dim, hierarchy_levels=3):

        super().__init__()

        self.hierarchy_levels = hierarchy_levels

        self.in_dim = in_dim

        

        # 分层处理模块

        self.level_processors = nn.ModuleList()

        

        for level in range(hierarchy_levels):

            # 每层维度递减

            next_dim = max(32, in_dim // (2 ** level))

            

            processor = nn.ModuleDict({

                'project_down': nn.Conv2d(in_dim, next_dim, 1),  # 修复：移除冗余逻辑

                'mona_op': MonaOp(next_dim),

                'project_up': nn.Conv2d(next_dim, in_dim, 1),

                'norm': LayerNorm2d(in_dim)

            })

            

            self.level_processors.append(processor)

        

        # 层间融合

        self.level_fusion = nn.ModuleList([

            nn.Conv2d(in_dim * 2, in_dim, 1) for _ in range(hierarchy_levels - 1)

        ])

        

        # 最终融合权重

        self.final_weights = nn.Parameter(torch.ones(hierarchy_levels) / hierarchy_levels)

        self.gamma = nn.Parameter(torch.ones(in_dim, 1, 1) * 1e-6)



    def forward(self, x):

        identity = x

        level_outputs = []

        current_input = x

        

        for level, processor in enumerate(self.level_processors):

            # 下投影

            projected = processor['project_down'](current_input)

            

            # Mona操作

            enhanced = processor['mona_op'](projected)

            

            # 上投影

            output = processor['project_up'](enhanced)

            output = processor['norm'](output)

            

            level_outputs.append(output)

            

            # 如果不是最后一层，进行层间融合

            if level < len(self.level_processors) - 1:

                fused = torch.cat([current_input, output], dim=1)

                current_input = self.level_fusion[level](fused)

        

        # 加权融合所有层的输出

        weighted_sum = sum(w * out for w, out in zip(self.final_weights, level_outputs))

        

        return identity + weighted_sum * self.gamma


class AdaptiveTSSA_Enhanced(nn.Module):

    """修复版：正确的输入形状处理"""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, 

                 scales=[1, 2, 4], hierarchy_levels=3):

        super().__init__()

        self.c = c

        self.add = shortcut

        

        # 修复：移除channels_last参数

        self.dyt1 = AdaptiveDynamicTanh(c, num_scales=len(scales))

        self.dyt2 = AdaptiveDynamicTanh(c, num_scales=len(scales))

        

        # 跨尺度注意力

        self.attn = CrossScaleAttentionTSSA(c, num_heads=num_heads, scales=scales)

        

        # 分层Mona模块

        self.mona1 = HierarchicalMona(c, hierarchy_levels=hierarchy_levels)

        self.mona2 = HierarchicalMona(c, hierarchy_levels=hierarchy_levels)

        

        # 增强的EDFFN

        self.ffn = EDFFN(c, ffn_expansion_factor=2, bias=False)

        

        # 特征门控

        self.feature_gate = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c, c // 4, 1),

            nn.ReLU(inplace=True),

            nn.Conv2d(c // 4, c, 1),

            nn.Sigmoid()

        )



    def forward(self, x):

        BS, C, H, W = x.size()

        

        # 第一阶段：自适应注意力

        x_dyt1 = self.dyt1(x)

        # 修复：传入4D张量而不是3D张量

        attn_out = self.attn(x_dyt1)  # 直接传入4D张量

        attn_out = attn_out.permute(0, 2, 1).view(-1, C, H, W).contiguous()

        

        if self.add:

            x = x + attn_out

        else:

            x = attn_out

            

        # 分层特征增强

        x = self.mona1(x)

        

        # 第二阶段：自适应FFN

        x_dyt2 = self.dyt2(x)

        ffn_out = self.ffn(x_dyt2)

        

        # 特征门控

        gate = self.feature_gate(x)

        ffn_out = ffn_out * gate

        

        if self.add:

            x = x + ffn_out

        else:

            x = ffn_out

            

        # 最终增强

        x = self.mona2(x)

        

        return x



class C2AdaptiveTSSA_Enhanced(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(
            AdaptiveTSSA_Enhanced(
                self.c, 
                attn_ratio=0.5, 
                num_heads=max(1, self.c // 64),
                scales=[1, 2, 4],
                hierarchy_levels=3
            ) for _ in range(n)
        ))
#C2PTSSA

###################################

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    """
    def __init__(self, c1, r=16):
        super().__init__()
        c_ = int(c1 / r)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, c1, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.avgpool(x))

class StandardFFN(nn.Module):
    """
    Standard FFN (Feed-Forward Network).
    """
    def __init__(self, c1, expansion=2, bias=False):
        super().__init__()
        c_ = int(c1 * expansion)
        self.cv1 = nn.Conv2d(c1, c_, 1, bias=bias)
        self.act = nn.GELU()
        self.cv2 = nn.Conv2d(c_, c1, 1, bias=bias)
    
    def forward(self, x):
        return self.cv2(self.act(self.cv1(x)))

class SimpleFeatureProcessor(nn.Module):
    """
    Simple Feature Processor (Norm + DW + PW).
    """
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=max(1, c // 32), num_channels=c)
        self.conv_dw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.act = nn.GELU()
        self.conv_pw = nn.Conv2d(c, c, 1) 
    
    def forward(self, x):
        x = self.norm(x)
        x = self.conv_dw(x)
        x = self.act(x)
        x = self.conv_pw(x)
        return x

class SFABlock(nn.Module):
    """
    SFABlock (Squeeze Feature Attention Block).
    Structure: [Pre-SE] -> [Pre-FFN]
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.c = c
        self.add = shortcut
        
        # 1. Attention Stage (SE)
        self.pre_attn = SimpleFeatureProcessor(c)
        self.attn = SEBlock(c) 
        

        self.pre_ffn = SimpleFeatureProcessor(c)
        self.ffn = StandardFFN(c, expansion=2, bias=False)
        
   
        self.res_w1 = nn.Parameter(torch.tensor(0.1))
        self.res_w2 = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        identity = x
        x_proc = self.pre_attn(x)
        attn_out = self.attn(x_proc)
        x = identity + attn_out * self.res_w1 if self.add else attn_out
        
        identity = x
        x_proc = self.pre_ffn(x)
        ffn_out = self.ffn(x_proc)
        x = identity + ffn_out * self.res_w2 if self.add else ffn_out
        
        return x

# class C2SFA(C2PSA):
#     """
#     C2SFA: C2 block with Squeeze Feature Attention.
#     类似于 C2PSA，但使用更现代的 SE+FFN 结构。
#     """
#     def __init__(self, c1, c2, n=1, e=0.5):
#         super().__init__(c1, c2, n, e)
#         self.m = nn.Sequential(*(
#             SFABlock(self.c, attn_ratio=0.5, num_heads=max(1, self.c // 64)) 
#             for _ in range(n)
#         ))



class ProgressiveTSSA_Fusion0(nn.Module):
    """
    SIMPLIFIED VERSION of ProgressiveTSSA_Fusion.
    复杂的自定义模块已被标准模块替换。
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.c = c
        self.add = shortcut
        
        # 1. 预处理 
        self.pre_attn_block = SimpleFeatureProcessor(c)
        
        # 2. 注意力 

        self.attn = SEBlock(c) 
        
        # 3. 简化的预处理
        self.pre_ffn_block = SimpleFeatureProcessor(c)
        
        # 4. 简化的 FFN 
        self.ffn = StandardFFN(c, expansion=2, bias=False)
        
        # 5. 残差权重
        self.residual_weight1 = nn.Parameter(torch.tensor(0.1))
        self.residual_weight2 = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        BS, C, H, W = x.size()
        
        # --- 阶段 1: Attention ---
        identity1 = x
        
        # 1a. 预处理 
        x_proc1 = self.pre_attn_block(x)
        
        # 1b. 注意力 
        # SEBlock 直接输出 4D 张量 (B,C,H,W)，无需 permute
        attn_out = self.attn(x_proc1)
        
        # 1c. 残差连接
        x = identity1 + attn_out * self.residual_weight1 if self.add else attn_out
        
        # --- 阶段 2: FFN ---
        identity2 = x
        
        # 2a. 预处理 
        x_proc2 = self.pre_ffn_block(x)
        
        # 2b. FFN
        ffn_out = self.ffn(x_proc2)
        
        # 2c. 残差连接
        x = identity2 + ffn_out * self.residual_weight2 if self.add else ffn_out
        
        return x

####################################

class ProgressiveFeatureFusion1(nn.Module):
    """渐进式特征融合模块"""
    def __init__(self, dim, num_stages=3):
        super().__init__()
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            stage = nn.ModuleDict({
                'conv': nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                'norm': nn.BatchNorm2d(dim),
                'activation': nn.GELU(),
                'channel_mix': nn.Conv2d(dim, dim, 1),
                'spatial_mix': nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
            })
            self.stages.append(stage)
        self.stage_fusion = nn.ModuleList([nn.Conv2d(dim * 2, dim, 1) for _ in range(num_stages - 1)])
        self.stage_attention = nn.Parameter(torch.ones(num_stages) / num_stages)

    def forward(self, x):
        stage_outputs = []
        current = x
        for i, stage in enumerate(self.stages):
            processed = stage['activation'](stage['norm'](stage['conv'](current)))
            out = stage['channel_mix'](processed) + stage['spatial_mix'](processed) + current
            stage_outputs.append(out)
            if i < len(self.stages) - 1:
                current = self.stage_fusion[i](torch.cat([current, out], dim=1))
        return sum(w * out for w, out in zip(self.stage_attention, stage_outputs)) + x

# class CrossScaleAttentionTSSA(nn.Module):
#     """多尺度 TSSA 注意力"""
#     def __init__(self, dim, num_heads=8, scales=[1, 2, 4]):
#         super().__init__()
#         self.heads = num_heads
#         self.scales = scales
#         self.qkv_projections = nn.ModuleList([nn.Linear(dim, dim * 3) for _ in scales])
#         self.cross_scale_fusion = nn.MultiheadAttention(dim, num_heads, batch_first=True)
#         self.temps = nn.Parameter(torch.ones(len(scales), num_heads, 1))
#         self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.))

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_flat = x.flatten(2).permute(0, 2, 1)
#         scale_features = []
#         for i, (scale, qkv_proj) in enumerate(zip(self.scales, self.qkv_projections)):
#             if scale > 1:
#                 x_s = F.interpolate(x, scale_factor=1/scale, mode='bilinear').flatten(2).permute(0, 2, 1)
#                 x_s = F.interpolate(x_s.permute(0,2,1).view(B,C,H//scale,W//scale), size=(H,W), mode='bilinear').flatten(2).permute(0,2,1)
#             else:
#                 x_s = x_flat
            
#             qkv = qkv_proj(x_s)
#             q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv.chunk(3, dim=-1))
#             attn = F.softmax(torch.sum(F.normalize(q, dim=-1)**2, dim=-1) * self.temps[i], dim=-1)
#             dots = torch.matmul(attn.unsqueeze(-2), k**2)
#             out = rearrange(-torch.mul(v.mul(attn.unsqueeze(-1)), 1./(1+dots)), 'b h n d -> b n (h d)')
#             scale_features.append(out)

#         if len(scale_features) > 1:
#             stacked = torch.stack(scale_features, dim=1).view(B, len(self.scales)*H*W, C)
#             fused, _ = self.cross_scale_fusion(stacked, stacked, stacked)
#             fused = fused.view(B, len(self.scales), H*W, C).mean(dim=1)
#         else:
#             fused = scale_features[0]
#         return self.to_out(fused)

class Mlp(nn.Module):
    """标准 MLP"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        return self.fc2(self.act(self.dwconv(self.fc1(x))))

class ProgressiveTSSA_Fusion1(nn.Module):
    """
    重构后的单个 TSSA 融合块 (增强版)
    变更：Stage 2 的 BatchNorm2d 被替换为 ProgressiveFeatureFusion
    """
    def __init__(self, c, num_heads=4, shortcut=True):
        super().__init__()
        self.c = c
        self.add = shortcut
        
        # Stage 1 组件
        self.feature_enhancement1 = ProgressiveFeatureFusion(c, num_stages=3)
        self.attn_norm = nn.GroupNorm(1, c)
        self.attn = CrossScaleAttentionTSSA(c, num_heads=num_heads, scales=[1, 2, 4])
        
        # Stage 2 组件
        # [修改点] 这里原本是 self.ffn_norm = nn.BatchNorm2d(c)
        # 现在替换为第二个渐进式特征融合模块
        self.feature_enhancement2 = ProgressiveFeatureFusion(c, num_stages=3)
        
        # self.ffn = Mlp(c, hidden_features=c*4)
        self.ffn = EDFFN(dim=c, ffn_expansion_factor=4)
        self.res_w1 = nn.Parameter(torch.tensor(0.1))
        self.res_w2 = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, C, H, W = x.shape
        
        # ------------------------------------------------
        # 1. Stage 1: PFF + Attention
        # ------------------------------------------------
        res1 = x
        # 先通过第一个特征增强
        x = self.feature_enhancement1(x)
        # 注意力机制
        attn_out = self.attn(self.attn_norm(x))
        attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=H, w=W)
        # 残差连接 1
        x = res1 + attn_out * self.res_w1 if self.add else attn_out
        
        # ------------------------------------------------
        # 2. Stage 2: PFF + FFN
        # ------------------------------------------------
        res2 = x
        # [修改点] 先通过第二个特征增强 (替代了原来的 BN)
        x = self.feature_enhancement2(x)
        # MLP 前馈
        ffn_out = self.ffn(x)
        # 残差连接 2
        x = res2 + ffn_out * self.res_w2 if self.add else ffn_out
        
        return x


class C2ProgressiveTSSA_Fusion1(C2PSA):
    """
    继承自 C2PSA，将内部的瓶颈层替换为 ProgressiveTSSA_Fusion。
    这种结构适合用于 YOLO 的 Head 或者 Backbone 尾部。
    """
    def __init__(self, c1, c2, n=1, e=0.5):
      
        super().__init__(c1, c2, n, e)
        

        self.m = nn.Sequential(*(
            ProgressiveTSSA_Fusion1(
                self.c, 
                num_heads=max(1, self.c // 32), # 假设每头32通道，避免维度不匹配
                shortcut=True
            ) for _ in range(n)
        ))


class C2SFA(C2PSA):
    """
    继承自 C2PSA，将内部的瓶颈层替换为 ProgressiveTSSA_Fusion。
    这种结构适合用于 YOLO 的 Head 或者 Backbone 尾部。
    """
    def __init__(self, c1, c2, n=1, e=0.5):

        super().__init__(c1, c2, n, e)
        

        self.m = nn.Sequential(*(
            ProgressiveTSSA_Fusion0(
                self.c, 
                num_heads=max(1, self.c // 64), # 假设每头32通道，避免维度不匹配
                shortcut=True
            ) for _ in range(n)
        ))
###################################
class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x_dtype = x.dtype
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8
        
        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        
        x=x[:,:,:h,:w]
        
        return x.to(x_dtype)

class CrossScaleAttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., 
                 scales=[1, 2, 4], **kwargs):
        super().__init__()
        self.heads = num_heads
        self.scales = scales
        self.dim = dim
        self.head_dim = dim // num_heads
        # 多尺度QKV投影
        self.qkv_projections = nn.ModuleList([

            nn.Linear(dim, dim * 3, bias=qkv_bias) for _ in scales

        ])
        # 跨尺度融合
        self.cross_scale_fusion = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True
        )
        # 温度参数 - 每个尺度独立
        self.temps = nn.Parameter(torch.ones(len(scales), num_heads, 1))
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        """修复：正确处理4D输入张量"""
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        scale_features = []
        # 多尺度特征提取
        for i, (scale, qkv_proj) in enumerate(zip(self.scales, self.qkv_projections)):
            if scale > 1:
                # 下采样
                x_scaled = F.adaptive_avg_pool2d(x, (H//scale, W//scale))
                x_scaled = x_scaled.flatten(2).permute(0, 2, 1)
                # 上采样回原尺寸
                x_scaled = F.interpolate(
                    x_scaled.permute(0, 2, 1).view(B, C, H//scale, W//scale),
                    size=(H, W), mode='bilinear', align_corners=False
                ).flatten(2).permute(0, 2, 1)
            else:
                x_scaled = x_flat
            # QKV计算
            qkv = qkv_proj(x_scaled)
            q, k, v = qkv.chunk(3, dim=-1)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
            # 修复：标准化维度改为dim=-1（特征维度）
            w_normed = F.normalize(q, dim=-1)
            w_sq = w_normed ** 2
            Pi = F.softmax(torch.sum(w_sq, dim=-1) * self.temps[i], dim=-1)
            # 修复：移除冗余的归一化计算
            dots = torch.matmul(Pi.unsqueeze(-2), k ** 2)
            attn = 1. / (1 + dots)
            attn = self.attn_drop(attn)
            out = -torch.mul(v.mul(Pi.unsqueeze(-1)), attn)
            out = rearrange(out, 'b h n d -> b n (h d)')
            scale_features.append(out)
        # 跨尺度特征融合
        if len(scale_features) > 1:
            stacked_features = torch.stack(scale_features, dim=1)  # [B, num_scales, HW, C]
            B, num_scales, HW, C = stacked_features.shape
            stacked_features = stacked_features.view(B, num_scales * HW, C)
            # 使用多头注意力进行跨尺度融合
            fused_features, _ = self.cross_scale_fusion(
                stacked_features, stacked_features, stacked_features
            )
            # 重新整形并平均
            fused_features = fused_features.view(B, num_scales, HW, C).mean(dim=1)
        else:
            fused_features = scale_features[0]
        return self.to_out(fused_features)
    
class AdaptiveDynamicTanh(nn.Module):

    """修复版：移除channels_last参数，专门支持channels_first"""

    def __init__(self, normalized_shape, num_scales=3):

        super().__init__()

        self.normalized_shape = normalized_shape

        self.num_scales = num_scales

        

        # 多尺度alpha参数

        self.alphas = nn.Parameter(torch.linspace(0.3, 1.0, num_scales).view(1, num_scales, 1, 1))

        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        

        # 权重和偏置 - 只支持channels_first

        self.weight = nn.Parameter(torch.ones(normalized_shape))

        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        

        # 特征重要性学习

        self.importance_gate = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(normalized_shape, normalized_shape // 4, 1),

            nn.ReLU(inplace=True),

            nn.Conv2d(normalized_shape // 4, num_scales, 1),

            nn.Softmax(dim=1)

        )



    def forward(self, x):

        # 计算特征重要性权重

        importance = self.importance_gate(x)  # [B, num_scales, 1, 1]

        

        # 多尺度激活

        multi_scale_outputs = []

        for i in range(self.num_scales):

            alpha_i = self.alphas[:, i:i+1, :, :]

            activated = torch.tanh(alpha_i * x)

            weighted = activated * importance[:, i:i+1, :, :]

            multi_scale_outputs.append(weighted)

        

        # 融合多尺度特征

        x = sum(multi_scale_outputs)

        

        # 应用权重和偏置 - channels_first格式

        x = x * self.weight[:, None, None] + self.bias[:, None, None]

        

        return x

class ProgressiveFeatureFusion(nn.Module):
    """渐进式特征融合模块"""
    def __init__(self, dim, num_stages=3):
        super().__init__()
        self.num_stages = num_stages
        
        # 渐进式处理阶段
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            stage = nn.ModuleDict({
                'conv': nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                'norm': nn.BatchNorm2d(dim),
                'activation': nn.GELU(),
                'channel_mix': nn.Conv2d(dim, dim, 1),
                'spatial_mix': nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
            })
            self.stages.append(stage)
        
        # 阶段间融合
        self.stage_fusion = nn.ModuleList([
            nn.Conv2d(dim * 2, dim, 1) for _ in range(num_stages - 1)
        ])
        
        # 注意力权重
        self.stage_attention = nn.Parameter(torch.ones(num_stages) / num_stages)

    def forward(self, x):
        stage_outputs = []
        current = x
        
        for i, stage in enumerate(self.stages):
            # 阶段处理
            processed = stage['conv'](current)
            processed = stage['norm'](processed)
            processed = stage['activation'](processed)
            
            # 通道和空间混合
            channel_mixed = stage['channel_mix'](processed)
            spatial_mixed = stage['spatial_mix'](processed)
            output = channel_mixed + spatial_mixed + current
            
            stage_outputs.append(output)
            
            # 如果不是最后阶段，进行融合
            if i < len(self.stages) - 1:
                fused = torch.cat([current, output], dim=1)
                current = self.stage_fusion[i](fused)
        
        # 加权融合所有阶段输出
        final_output = sum(w * out for w, out in zip(self.stage_attention, stage_outputs))
        
        return final_output + x

class ProgressiveTSSA_Fusion(nn.Module):


    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):

        super().__init__()

        self.c = c

        self.add = shortcut

        

        # 渐进式特征融合
        self.progressive_fusion1 = ProgressiveFeatureFusion(c, num_stages=3)
        self.progressive_fusion2 = ProgressiveFeatureFusion(c, num_stages=3)
        self.dyt1 = AdaptiveDynamicTanh(c, num_scales=3)
        self.dyt2 = AdaptiveDynamicTanh(c, num_scales=3)

        self.attn = CrossScaleAttentionTSSA(c, num_heads=num_heads, scales=[1, 2, 4])
        self.ffn = EDFFN(c, ffn_expansion_factor=2, bias=False)
        # 残差学习权重

        self.residual_weight1 = nn.Parameter(torch.tensor(0.1))

        self.residual_weight2 = nn.Parameter(torch.tensor(0.1))



    def forward(self, x):

        BS, C, H, W = x.size()

        identity = x

        

        # 第一阶段：渐进式处理 + 注意力

        x = self.progressive_fusion1(x)

        x_dyt1 = self.dyt1(x)

        # 修复：传入4D张量

        attn_out = self.attn(x_dyt1)

        attn_out = attn_out.permute(0, 2, 1).view(-1, C, H, W).contiguous()

        

        x = identity + attn_out * self.residual_weight1 if self.add else attn_out

        

        # 第二阶段：渐进式处理 + FFN

        x = self.progressive_fusion2(x)

        x_dyt2 = self.dyt2(x)

        ffn_out = self.ffn(x_dyt2)
        x = x + ffn_out * self.residual_weight2 if self.add else ffn_out

        

        return x

class C2ProgressiveTSSA_Fusion(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(
            ProgressiveTSSA_Fusion(
                self.c, 
                attn_ratio=0.5, 
                num_heads=max(1, self.c // 64)
            ) for _ in range(n)
        ))
C2PTSSA=C2ProgressiveTSSA_Fusion
# C2TSSA_DYT_Mona_EDFFN = C2SF

# C2TSSA_DYT_Mona_EDFFN=C2ProgressiveTSSA_Fusion