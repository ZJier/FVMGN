import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from WTConv import WTConv2d, create_wavelet_filter, _ScaleModule, wavelet_transform, inverse_wavelet_transform


# ================================================================================
# ================================Original ViT====================================
# ================================================================================
class Attention_ori(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, num_patch: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.attn = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        if num_heads == 1 and dim == head_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(self.inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)
        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        scores = scores * self.scale
        attn = self.attn(scores)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Transformer_ori(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, head_dim: int, hidden_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                nn.Sequential(nn.LayerNorm(dim), Attention_ori(dim, head_dim, num_heads, num_patch, patch_size)),
                nn.Sequential(nn.LayerNorm(dim), FeedForward(dim, hidden_dim))
            ]
            self.layers.append(nn.ModuleList(layer))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ================================================================================
# ====================================CTMixer=====================================
# ================================================================================
class MHSAC(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, num_patch: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads

        self.scale = head_dim ** -0.5
        self.attn = nn.Softmax(dim=-1)

        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)
        self.qkvc = nn.Conv2d(dim, self.inner_dim * 4, kernel_size=1, padding=0, groups=dim, bias=False)
        self.bnc = nn.BatchNorm2d(self.inner_dim)
        self.local = nn.Conv2d(self.inner_dim, self.inner_dim, kernel_size=3, padding=1, groups=self.inner_dim, bias=False)
        self.avgpool=nn.AdaptiveAvgPool1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)

        qkvc = self.qkvc(self.act(self.bn(x)))
        qkvc = qkvc.contiguous().view(b, self.num_patch*self.num_patch, self.inner_dim * 4)
        qkvc = qkvc.chunk(4, dim=-1)
        q, k, v, c = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkvc)

        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        scores = scores * self.scale
        attn = self.attn(scores)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        
        c = self.local(self.act(self.bnc(c.reshape(b, self.inner_dim, self.num_patch, self.num_patch)))).reshape(b, n, -1)

        out = self.avgpool(out + c)

        return out


class Convs(nn.Module):
    def __init__(self, dim: int, num_patch: int, hidden_dim: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.num_patch = num_patch
        self.patch_size = patch_size

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(dim), nn.GELU(), 
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0, bias=False))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim), nn.GELU(), 
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False))
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim), nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, kernel_size=1, padding=0, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _ = x.shape
        x_reshape = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)
        out1 = self.conv1(x_reshape)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2) + x_reshape
        result = out3.contiguous().view(b, self.num_patch * self.num_patch, self.dim)
        return result


class CTMixer(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, head_dim: int, hidden_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                nn.Sequential(nn.LayerNorm(dim), MHSAC(dim, head_dim, num_heads, num_patch, patch_size)),
                nn.Sequential(nn.LayerNorm(dim), Convs(dim, num_patch, hidden_dim, patch_size))
            ]
            self.layers.append(nn.ModuleList(layer))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ================================================================================
# ===================================WCTMixer=====================================
# ================================================================================
class WTAttn(nn.Module):
    def __init__(self, dim, num_heads, head_dim, wt_levels=1, wt_type='db1'):
        super(WTAttn, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads
        self.attn = nn.Softmax(dim=-1)
        self.wt_levels = wt_levels
        self.scale = head_dim ** -0.5

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, dim, dim, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(self.dim)
        self.qkvc = nn.Conv2d(self.dim, self.inner_dim * 4, kernel_size=1, padding=0, groups=self.dim, bias=False)
        self.bnc = nn.BatchNorm2d(self.inner_dim)
        self.local = nn.Conv2d(self.inner_dim, self.inner_dim, kernel_size=3, padding=1, groups=self.inner_dim, bias=False)
        self.avgpool=nn.AdaptiveAvgPool1d(self.dim)

        self.wavelet_scale = nn.ModuleList([_ScaleModule([1, dim, 1, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)])
    
    def freq_attn(self, input_freq, b, h): 
        curr_qkvc = self.qkvc(self.act(self.bn(input_freq)))
        if h % 2 == 0: 
            curr_qkvc = curr_qkvc.contiguous().view(b, (h//2)*(h//2), -1)
        else: 
            curr_qkvc = curr_qkvc.contiguous().view(b, (h//2+1)*(h//2+1), -1)
        curr_x_qkvc = curr_qkvc.chunk(4, dim=-1)
        q, k, v, c = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), curr_x_qkvc)
        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        scores = scores * self.scale
        attn = self.attn(scores)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        if h % 2 == 0: 
            c = self.local(self.act(self.bnc(c.reshape(b, self.inner_dim, h//2, h//2)))).reshape(b, -1, self.inner_dim)
            out_all = self.avgpool(out + c).reshape(b, self.dim, 1, h//2, h//2)
        else: 
            c = self.local(self.act(self.bnc(c.reshape(b, self.inner_dim, h//2+1, h//2+1)))).reshape(b, -1, self.inner_dim)
            out_all = self.avgpool(out + c).reshape(b, self.dim, 1, h//2+1, h//2+1)
        return out_all

    def forward(self, x):
        b, n, c = x.shape
        h = int(np.sqrt(n))
        x = x.contiguous().view(b, c, h, h)
        
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels): 
            h = curr_x_ll.shape[2]
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0:1, :, :].squeeze(dim=2)
            curr_x_lh = curr_x[:, :, 1:2, :, :].squeeze(dim=2)
            curr_x_hl = curr_x[:, :, 2:3, :, :].squeeze(dim=2)
            curr_x_hh = curr_x[:, :, 3:4, :, :].squeeze(dim=2)
            # =============================================================================
            out_ll_all = self.wavelet_scale[i](self.freq_attn(curr_x_ll, b, h)).squeeze(dim=2)
            out_lh_all = self.wavelet_scale[i](self.freq_attn(curr_x_lh, b, h))
            out_hl_all = self.wavelet_scale[i](self.freq_attn(curr_x_hl, b, h))
            out_hh_all = self.wavelet_scale[i](self.freq_attn(curr_x_hh, b, h))
            out_h_all = torch.cat([out_lh_all, out_hl_all, out_hh_all], dim=2)
            # =============================================================================
            x_ll_in_levels.append(out_ll_all)
            x_h_in_levels.append(out_h_all)
        
        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1): 
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
        
        x_tag = next_x_ll.reshape(b, n, c)
        return x_tag


class WConvs(nn.Module):
    def __init__(self, dim): 
        super().__init__()
        self.wtconv = WTConv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, bias=True, wt_levels=2, wt_type='db1')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, hw, dim = x.shape
        h = int(np.sqrt(hw))
        x_reshape = x.contiguous().view(b, dim, h, h)
        out = self.wtconv(x_reshape)
        result = out.contiguous().view(b, h * h, dim)
        return result


class WCTMixer(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, head_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                nn.Sequential(nn.LayerNorm(dim), WTAttn(dim, num_heads, head_dim, wt_levels=1, wt_type='db1')),
                nn.Sequential(nn.LayerNorm(dim), WConvs(dim))
            ]
            self.layers.append(nn.ModuleList(layer))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



