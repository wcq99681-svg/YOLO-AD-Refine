import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from typing import List, Optional, Callable, Union, Tuple
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from functools import partial # Needed for LayerNormGeneral if used

# Imports provided by the user (assuming these paths are valid in the target environment)
from ultralytics.utils.tal import dist2bbox, make_anchors # Keeping only used imports
from .block import DFL # Importing DFL from .block as requested
from .conv import Conv, DWConv # Importing Conv, DWConv from .conv as requested
# Removed unused imports like BNContrastiveHead, Proto, MLP, DeformableTransformer etc.
# Removed unused init functions like constant_, xavier_uniform_
# Removed unused utils like bias_init_with_prob, linear_init
# Removed unused traceback

# __all__ list provided by the user (kept for module structure context)
__all__ = ["Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect", "Detect_Aphid_Light"] # Updated class name

# --- Utility Functions (still needed) ---
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6

# --- EMA Module (User Provided Version) ---
# Definition remains here as it's a new/custom module for this context
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        if channels == 0: # Handle zero channels case
             self.groups = 1
             group_channels = 0
             #print(f"Warning: EMA input channels is 0.") # Reduce verbosity
        elif channels // self.groups <= 0:
             #print(f"Warning: EMA channels ({channels}) <= factor ({factor}). Adjusting groups to 1.") # Reduce verbosity
             self.groups = 1
             group_channels = channels // self.groups
        else:
             group_channels = channels // self.groups

        # Initialize layers only if group_channels > 0
        if group_channels > 0:
            self.softmax = nn.Softmax(-1)
            self.agp = nn.AdaptiveAvgPool2d((1, 1))
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))
            self.gn = nn.GroupNorm(group_channels, group_channels)
            self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1, stride=1, padding=0)
            self.conv3x3 = nn.Conv2d(group_channels, group_channels, kernel_size=3, stride=1, padding=1)
        else:
            # If channels are 0, make forward an identity operation
            self.forward = lambda x: x


    def forward(self, x):
        b, c, h, w = x.size()
        if c == 0: return x # Handle zero channels case in forward pass

        group_channels = c // self.groups
        # Ensure group_channels is calculated correctly based on actual input 'c'
        if c // self.groups <= 0:
            # Fallback if dynamic shape causes issues, should ideally not happen if init logic is sound
            #print(f"Warning: EMA forward pass with c={c}, groups={self.groups}. Adjusting groups to 1.") # Reduce verbosity
            self.groups = 1
            group_channels = c

        group_x = x.reshape(b * self.groups, group_channels, h, w)

        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, group_channels, -1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, group_channels, -1)

        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, group_channels, -1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, group_channels, -1)

        weights1 = torch.matmul(x11, x12)
        weights2 = torch.matmul(x21, x22)
        weights = (weights1 + weights2).reshape(b * self.groups, 1, h, w)

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# --- PartialConv Module (from FasterNet) ---
# Definition remains here as it's a new/custom module for this context
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        actual_conv_dim = max(1, self.dim_conv3)

        if dim == 0: # Handle zero dim case
            self.dim_conv3 = 0
            self.dim_untouched = 0
            self.partial_conv3 = nn.Identity()
            #print("Warning: Partial_conv3 input dim is 0.") # Reduce verbosity
        elif self.dim_conv3 == 0:
             self.dim_untouched = dim
             self.partial_conv3 = nn.Identity() # Use Identity if no channels for conv
             #print(f"Warning: Partial_conv3 input dim {dim} too small for n_div {n_div}. Using Identity.")
        else:
            self.partial_conv3 = nn.Conv2d(actual_conv_dim, actual_conv_dim, 3, 1, 1, bias=False)
            if actual_conv_dim != self.dim_conv3:
                #print(f"Warning: Adjusted partial conv dim from {self.dim_conv3} to {actual_conv_dim}")
                self.dim_conv3 = actual_conv_dim # Update split size if adjusted

        self.forward_mode = forward

    def forward(self, x):
        if self.dim_conv3 == 0 or isinstance(self.partial_conv3, nn.Identity):
            return x

        if self.forward_mode == 'split_cat':
            # Ensure dim_untouched calculation uses the potentially adjusted dim_conv3
            actual_dim_untouched = x.size(1) - self.dim_conv3
            if actual_dim_untouched < 0:
                print(f"Error: Negative dimension in split: {self.dim_conv3}, {actual_dim_untouched}, input channels {x.size(1)}")
                return x
            try:
                x1, x2 = torch.split(x, [self.dim_conv3, actual_dim_untouched], dim=1)
            except RuntimeError as e:
                print(f"Error during torch.split: {e}")
                print(f"Input shape: {x.shape}, split sizes: [{self.dim_conv3}, {actual_dim_untouched}]")
                return x # Fallback
            x1 = self.partial_conv3(x1)
            x = torch.cat((x1, x2), 1)
            return x
        elif self.forward_mode == 'slicing':
            x = x.clone()
            x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
            return x
        else:
            raise NotImplementedError


# --- GSConv Module (from SlimNeck) ---
# Definition remains here as it's a new/custom module for this context
class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        if c_ == 0 and c2 > 0:
            #print(f"Warning: GSConv c2 ({c2}) too small, adjusting intermediate channels.") # Reduce verbosity
            c_ = 1
            self.cv1 = Conv(c1, c_, k, s, p, g, d, act=Conv.default_act)
            self.cv2 = Conv(c_, max(0, c2 - c_), 5, 1, p, 1, d, act=Conv.default_act) # DWConv part uses groups=1 here
            self._split_indices = [c_, max(0, c2 - c_)]
        elif c2 == 0:
             #print(f"Warning: GSConv c2 is 0. Module might not function as expected.") # Reduce verbosity
             self.cv1 = nn.Identity()
             self.cv2 = nn.Identity()
             self._split_indices = [0, 0]
        else:
            self.cv1 = Conv(c1, c_, k, s, p, g, d, act=Conv.default_act)
            self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, act=Conv.default_act) # DWConv part
            self._split_indices = [c_, c_]

    def forward(self, x):
        if isinstance(self.cv1, nn.Identity):
            return x

        x1 = self.cv1(x)
        x2 = self.cv2(x1)

        tensors_to_cat = []
        # Check actual output channels before appending
        if x1.size(1) > 0: tensors_to_cat.append(x1)
        if x2.size(1) > 0: tensors_to_cat.append(x2)

        if not tensors_to_cat:
             b, _, h, w = x.shape
             return torch.zeros((b, 0, h, w), device=x.device, dtype=x.dtype)

        x_cat = torch.cat(tensors_to_cat, 1)

        b, n, h, w = x_cat.size()
        if n == 0: return x_cat

        # Perform shuffle only if channel number is even and > 0
        if n > 1 and n % 2 == 0:
            n_half = n // 2
            x_cat = x_cat.view(b, 2, n_half, h, w)
            x_cat = x_cat.permute(0, 2, 1, 3, 4).contiguous()
            y = x_cat.view(b, n, h, w)
            return y
        else:
            #print(f"Warning: GSConv shuffle skipped due to odd or zero channels ({n}).") # Reduce verbosity
            return x_cat


# --- Light Attention Block ---
# Definition remains here as it's a new/custom module for this context
class LightAttnBlock(nn.Module):
    """ Lightweight Attention Block using PartialConv and EMA. """
    def __init__(self, in_channels, partial_n_div=4):
        super().__init__()
        self.spatial_mix = Partial_conv3(in_channels, n_div=partial_n_div)
        self.channel_mix = EMA(in_channels)

    def forward(self, x):
        spatial_feat = self.spatial_mix(x)
        channel_feat = self.channel_mix(spatial_feat)
        return channel_feat


# --- Innovative Detect Head for Aphids (Lightweight) ---
# Definition remains here as it's the core new module
class Detect_Aphid_Light(nn.Module):
    """ Lightweight YOLO head for aphid detection (Standalone). """
    dynamic = False
    export = False
    max_det = 300
    shape = None
    anchors = torch.empty(0) # Will be populated by _inference
    # stride_tensor = torch.empty(0) # Removed Class Attribute
    anchors_calculated = False # Flag to track if anchors/strides are calculated

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        # Register stride as buffer (stores unique strides like [8, 16, 32])
        self.register_buffer("stride", torch.zeros(self.nl), persistent=True)
        # ========= 修改部分开始 =========
        # 仅在 __init__ 中注册 stride_tensor buffer
        # Initialize with dummy size, will be resized in _inference
        self.register_buffer("stride_tensor", torch.empty(0), persistent=False)
        # ========= 修改部分结束 =========

        c1 = ch[0] if ch else 0
        c2 = max((16, c1 // 8 if c1 > 0 else 0, self.reg_max * 4))
        c3 = max(c1 // 2 if c1 > 0 else 0, min(self.nc, 100))

        # Ensure channels c2, c3 are not zero if input channels ch are non-zero
        if c1 > 0:
            c2 = max(1, c2)
            c3 = max(1, c3)

        self.enhancement_blocks = nn.ModuleList(LightAttnBlock(x) for x in ch)

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                GSConv(c2, c2, k=3),
                EMA(c2),
                nn.Conv2d(c2, 4 * self.reg_max, 1, bias=True)
            ) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                GSConv(c3, c3, k=3),
                EMA(c3),
                nn.Conv2d(c3, self.nc, 1, bias=True)
            ) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        enhanced_x = [self.enhancement_blocks[i](x[i]) for i in range(self.nl)]
        output_x = []
        for i in range(self.nl):
            pred_reg = self.cv2[i](enhanced_x[i])
            pred_cls = self.cv3[i](enhanced_x[i])
            output_x.append(torch.cat((pred_reg, pred_cls), 1))

        if self.training:
            return output_x

        y = self._inference(output_x)
        return y if self.export else (y, output_x)

    def _inference(self, x):
        shape = x[0].shape # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2) # Shape: (B, no, N)

        needs_calculation = not self.anchors_calculated or \
                           self.dynamic or \
                           self.shape != shape

        # Infer strides only if necessary and not already calculated/set
        if needs_calculation and hasattr(self, 'stride') and not torch.any(self.stride > 0):
             #print("\n!!! WARNING: Strides appear uninitialized (all zeros in buffer). " # Reduce verbosity
             #      "Attempting inference based on input feature map shapes assuming input size 640. "
             #      "This might be unreliable if input size varies. For robust behavior, ensure the model building process sets the `stride` attribute correctly.\n")
             try:
                  imgsz = 640 # Assume fixed input size
                  strides_inferred = torch.tensor([imgsz / xi.shape[-1] for xi in x], device=x[0].device, dtype=torch.float32)
                  if torch.any(strides_inferred <= 0) or len(strides_inferred) != self.nl:
                      raise ValueError(f"Inferred strides {strides_inferred} are invalid.")
                  self.stride.copy_(strides_inferred) # Update unique stride buffer
                  #print(f"Inferred and set strides: {self.stride.cpu().numpy().tolist()}") # Reduce verbosity
             except Exception as e:
                  print(f"ERROR: Failed to infer strides: {e}.")
                  raise ValueError("Strides required but not set or inferable.")

        # Calculate anchors and stride_tensor using make_anchors if needed
        if needs_calculation:
            try:
                if not hasattr(self, 'stride') or not torch.any(self.stride > 0):
                     raise ValueError("Cannot make anchors - strides are invalid.")
                device = x[0].device
                current_strides_for_make = self.stride.to(device) # Use the unique stride buffer

                # make_anchors returns anchor_points (N, 2) and stride_tensor (N, 1)
                anchor_points, stride_tensor = make_anchors(x, current_strides_for_make, 0.5)

                # ========= 修改部分开始 =========
                # 直接存储 make_anchors 的原始输出
                self.anchors = anchor_points # 形状 (N, 2)
                # 将 stride_tensor (N, 1) 存入 buffer
                if self.stride_tensor.shape != stride_tensor.shape:
                    self.register_buffer("stride_tensor", stride_tensor, persistent=False)
                else:
                    self.stride_tensor.copy_(stride_tensor)
                # ========= 修改部分结束 =========

                # Update self.stride buffer based on unique values in stride_tensor for consistency
                unique_strides = torch.unique(stride_tensor.squeeze()) # Squeeze to get unique values
                if len(unique_strides) == self.nl:
                     self.stride.copy_(unique_strides.to(self.stride.device))
                else:
                     print(f"Warning: Unexpected number of unique strides ({len(unique_strides)}) found in stride_tensor.")

                self.shape = shape
                self.anchors_calculated = True
            except Exception as e:
                 print(f"ERROR: Exception during make_anchors in _inference: {e}")
                 print(f"Input shapes: {[xi.shape for xi in x]}")
                 print(f"self.stride used: {self.stride}")
                 raise e

        # Ensure necessary tensors are calculated and on the right device
        if not hasattr(self, 'anchors') or not hasattr(self, 'stride_tensor') or not self.anchors_calculated:
             raise RuntimeError("Anchors or stride_tensor not calculated before decoding.")
        if self.anchors.device != x_cat.device: self.anchors = self.anchors.to(x_cat.device)
        if self.stride_tensor.device != x_cat.device: self.stride_tensor = self.stride_tensor.to(x_cat.device)

        # Decoding logic
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) # box shape (B, 4*reg_max, N)

        # ========= 修改部分开始 =========
        # 调整形状以匹配 decode_bboxes 和后续缩放
        anchors_for_decode = self.anchors.unsqueeze(0) # 形状 (1, N, 2)
        # stride_tensor 用于缩放，需要形状 (1, N, 1)
        strides_for_scaling = self.stride_tensor.unsqueeze(0) # 形状 (1, N, 1)
        # ========= 修改部分结束 =========

        # decode_bboxes 现在返回形状 (B, N, 4)
        dbox = self.decode_bboxes(box, anchors_for_decode) * strides_for_scaling # 应用缩放

        # ========= 修改部分开始 =========
        # 最终拼接需要形状 (B, N, 4) 和 (B, N, nc)
        y = torch.cat((dbox, cls.permute(0, 2, 1).sigmoid()), dim=-1) # cls: (B, nc, N) -> (B, N, nc)
        # 返回结果需要匹配原始格式 (B, 4+nc, N)
        return y.permute(0, 2, 1) # (B, N, 4+nc) -> (B, 4+nc, N)
        # ========= 修改部分结束 =========


    def bias_init(self):
        # Use self.stride which is now a buffer
        if hasattr(self, 'stride') and self.stride is not None and torch.any(self.stride > 0):
             # Rest of the bias_init logic remains mostly the same
             if len(self.stride) != self.nl:
                  print(f"Warning: Stride length ({len(self.stride)}) does not match number of layers ({self.nl}) during bias_init.")
             stride_iter = self.stride
             if len(self.stride) > self.nl:
                 stride_iter = self.stride[:self.nl]
             elif len(self.stride) < self.nl:
                  print(f"Error: Stride length ({len(self.stride)}) is less than number of layers ({self.nl}). Cannot perform bias_init correctly.")
                  return

             for a, b, s in zip(self.cv2, self.cv3, stride_iter):
                  if isinstance(a[-1], nn.Conv2d) and hasattr(a[-1], 'bias') and a[-1].bias is not None:
                       a[-1].bias.data[:] = 1.0
                  if isinstance(b[-1], nn.Conv2d) and hasattr(b[-1], 'bias') and b[-1].bias is not None:
                       s_item = s.item() if isinstance(s, torch.Tensor) else s
                       s_safe = max(s_item, 1e-6)
                       try:
                            safe_nc = max(self.nc, 1)
                            b[-1].bias.data[: self.nc] = math.log(5 / safe_nc / (640 / s_safe) ** 2)
                       except ValueError as e:
                            print(f"Warning: math.log error during bias_init for cls with stride {s_safe}: {e}")
                            b[-1].bias.data[: self.nc] = -4.5
        else:
             print("\n!!! WARNING: Strides not available during bias_init(). Biases will not be initialized. "
                   "Ensure the model building process sets the `stride` attribute OR rely on inference-time stride calculation.\n")

    # ========= 修改部分开始 =========
    # 修改 decode_bboxes 以匹配 _inference 中调整后的输入/输出形状
    def decode_bboxes(self, bboxes, anchors):
        # bboxes 形状 (B, 4*reg_max, N)
        # anchors 形状 (1, N, 2)
        decoded_dist = self.dfl(bboxes) # 输出形状 (B, 4, N)
        # dist2bbox 需要 dist=(B, N, 4), anchors=(1, N, 2)
        return dist2bbox(decoded_dist.permute(0, 2, 1), anchors, xywh=True, dim=-1) # 返回 (B, N, 4)
    # ========= 修改部分结束 =========


# --- Example Usage ---
# Example usage remains the same
if __name__ == '__main__':
    ch_example = (128, 256, 512)
    num_classes = 1 # aphids

    head_light = Detect_Aphid_Light(nc=num_classes, ch=ch_example)

    # --- NO MANUAL STRIDE SETTING NEEDED IN THIS EXAMPLE ---
    # We are relying on the internal inference logic now
    # print("Simulating external stride setting...")
    # head_light.stride = torch.tensor([8., 16., 32.])
    # print(f"Head stride set to: {head_light.stride.numpy().tolist()}")
    # --- END SIMULATION ---

    # Call bias_init - it might print a warning if stride isn't set yet, which is expected before the first inference
    head_light.bias_init()

    inputs = [
        torch.randn(2, ch_example[0], 80, 80), # P3 (stride 8)
        torch.randn(2, ch_example[1], 40, 40), # P4 (stride 16)
        torch.randn(2, ch_example[2], 20, 20)  # P5 (stride 32)
    ]

    print("\n--- Training Forward ---")
    head_light.train()
    outputs_train = head_light(inputs)
    print("Light Training outputs per level:")
    for i, out in enumerate(outputs_train):
        print(f" Level {i}: {out.shape}")

    print("\n--- Inference Forward (Stride Inference Attempt) ---")
    head_light.eval()
    head_light.export = False
    with torch.no_grad():
        try:
             # _inference will now attempt to infer strides if needed
             outputs_infer_decoded, outputs_infer_raw = head_light(inputs)
             print("\nLight Inference output (decoded):", outputs_infer_decoded.shape)
             print("Light Inference raw outputs per level:")
             for i, out in enumerate(outputs_infer_raw):
                  print(f" Level {i}: {out.shape}")
             print(f"Stride buffer value after inference: {head_light.stride.cpu().numpy().tolist()}") # Check stride buffer
             # print(f"Stride tensor buffer shape: {head_light.stride_tensor.shape}") # Check stride_tensor buffer
        except ValueError as e:
             print(f"\nERROR during inference: {e}")
             import traceback
             traceback.print_exc()
        except RuntimeError as e: # Catch potential RuntimeError from make_anchors
             print(f"\nRUNTIME ERROR during inference: {e}")
             import traceback
             traceback.print_exc()


    print("\n--- Export Inference Forward ---")
    head_light.export = True
    with torch.no_grad():
        try:
             outputs_export = head_light(inputs)
             print("\nLight Inference output (export mode):", outputs_export.shape)
        except ValueError as e:
             print(f"\nERROR during export inference: {e}")
             import traceback
             traceback.print_exc()
        except RuntimeError as e: # Catch potential RuntimeError from make_anchors
             print(f"\nRUNTIME ERROR during export inference: {e}")
             import traceback
             traceback.print_exc()

