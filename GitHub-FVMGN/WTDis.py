import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import cv2
import pywt.data
import numpy as np
from sklearn.decomposition import PCA
import random
# import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import scipy
from skimage import exposure


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x): 
        return torch.mul(self.weight, x)


class WTDs(nn.Module): 
    def __init__(self, in_channels, kernel_size=5, wt_levels=1, wt_type='db1', imgsize=11):
        super(WTDs, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.wt_levels = wt_levels
        self.wt_type = wt_type
        self.imgsize = imgsize
        self.eps = 1e-6

        self.wt_filter, self.iwt_filter = self.create_wavelet_filter(self.wt_type, self.in_channels, self.in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wavelet_convs = nn.ModuleList([nn.Conv2d(self.in_channels*4, self.in_channels*4, self.kernel_size, 
                                                      padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) 
                                                      for _ in range(self.wt_levels)])
        self.wavelet_scale = nn.ModuleList([_ScaleModule([1, self.in_channels*4, 1, 1], init_scale=0.1) 
                                            for _ in range(self.wt_levels)])
        
        self.base_conv = nn.Conv2d(2, 1, kernel_size=self.imgsize, padding=self.imgsize // 2, bias=False)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])
        self.sigmoid = nn.Sigmoid()
    
    def create_wavelet_filter(self, wave, in_size, out_size, type=torch.float):
        w = pywt.Wavelet(wave)

        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
        dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1), dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1), 
                                   dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1), dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
        dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

        rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
        rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
        rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1), rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1), 
                                   rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1), rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
        rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

        return dec_filters, rec_filters

    def wavelet_transform(self, x, filters): 
        b, c, h, w = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
        x = x.reshape(b, c, 4, h // 2, w // 2)
        return x

    def inverse_wavelet_transform(self, x, filters): 
        b, c, _, h_half, w_half = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        x = x.reshape(b, c * 4, h_half, w_half)
        x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
        return x

    def Multi_WTs(self, x):
        wt_in_levels = []
        wt_ll_in_levels = []
        wt_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x
        for i in range(self.wt_levels): 
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
            curr_x = self.wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]
            wt_in_levels.append(curr_x)
            wt_ll_in_levels.append(curr_x[:, :, 0, :, :])
            wt_h_in_levels.append(curr_x[:, :, 1:4, :, :])
        return wt_in_levels, wt_ll_in_levels, wt_h_in_levels, shapes_in_levels
    
    def WTinConv(self, wt_in_levels): 
        wtc_ll_in_levels = []
        wtc_h_in_levels = []
        for i in range(self.wt_levels): 
            curr_x_ll = wt_in_levels[i]
            curr_shape = curr_x_ll.shape
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
            shape_x = wt_in_levels[i].shape
            curr_x_tag = wt_in_levels[i].reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            wtc_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            wtc_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
        return wtc_ll_in_levels, wtc_h_in_levels
    
    def Multi_IWTs(self, x_ll_in_levels, x_h_in_levels, shapes_in_levels): 
        next_x_ll = 0
        x_tags = []
        for i in range(self.wt_levels-1, -1, -1): 
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll.squeeze(axis=2) + next_x_ll
            curr_x_ll = curr_x_ll.unsqueeze(axis=2)
            curr_x = torch.cat([curr_x_ll, curr_x_h], dim=2)
            next_x_ll = self.inverse_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
            x_tags.append(next_x_ll)
        return x_tags
    
    def guass_modeling(self, wt_ll_in_levels): 
        output_guass = []
        for i in range(self.wt_levels): 
            wt_ll_in_levelsi = wt_ll_in_levels[i]
            mean = wt_ll_in_levelsi.mean(dim=(2, 3), keepdim=True)
            var = ((wt_ll_in_levelsi - mean) ** 2).mean(dim=(2, 3), keepdim=True)
            std = (var + self.eps).sqrt()
            alpha = 0.1 # 0~1
            rho = torch.randn_like(std) * alpha
            # rho = torch.rand_like(std) * 2 - 1.
            outputs = wt_ll_in_levelsi + rho * std
            output_guass.append(outputs)
        return output_guass
    
    def hist_modeling(self, wt_h_in_levels):
        hist_h_all = []
        attn_hist_all = []
        for i in range(self.wt_levels): 
            hist_h = torch.sqrt((wt_h_in_levels[i][:, :, 0, :]**2 + wt_h_in_levels[i][:, :, 1, :]**2 + wt_h_in_levels[i][:, :, 2, :]**2) / 3)
            min_values = hist_h.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
            max_values = hist_h.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            hist_h = (hist_h - min_values) / (max_values - min_values)
            hist_h = torch.from_numpy(exposure.equalize_adapthist(hist_h.cpu().detach().numpy()))
            max_guass_hist, _ = torch.max(hist_h, dim=1, keepdim=True)
            avg_guass_hist = torch.mean(hist_h, dim=1, keepdim=True)
            spatial_attn_hist = self.sigmoid(self.base_scale(self.base_conv(torch.cat([max_guass_hist.cuda(), avg_guass_hist.cuda()], dim=1))))
            hist_h_all.append(hist_h)
            attn_hist_all.append(spatial_attn_hist.unsqueeze(axis=2))
        return hist_h_all, attn_hist_all
    
    def spat_attn(self, output_guass): 
        attn_matrix = []
        for i in range(self.wt_levels): 
            max_guass, _ = torch.max(output_guass[i], dim=1, keepdim=True)
            avg_guass = torch.mean(output_guass[i], dim=1, keepdim=True)
            spatial_attn = self.sigmoid(self.base_scale(self.base_conv(torch.cat([max_guass, avg_guass], dim=1))))
            spatial_attn = spatial_attn.unsqueeze(axis=2)
            attn_matrix.append(spatial_attn)
        return attn_matrix
    
    def cross_attn(self, spat_attn1, attn_hist1, wtc_ll_in_levels, wtc_h_in_levels, 
                   spat_attn2, attn_hist2, wtc_ll_in_levels2, wtc_h_in_levels2): 
        ll_x1, h_x1, ll_x2, h_x2 = [], [], [], []
        for i in range(self.wt_levels): 
            ll_attn = spat_attn1[i] * wtc_ll_in_levels2[i].unsqueeze(axis=2)
            hl_attn = attn_hist1[i] * wtc_h_in_levels2[i][:, :, 0:1, :, :]
            lh_attn = attn_hist1[i] * wtc_h_in_levels2[i][:, :, 1:2, :, :]
            hh_attn = attn_hist1[i] * wtc_h_in_levels2[i][:, :, 2:3, :, :]
            h_attn = torch.cat([hl_attn, lh_attn, hh_attn], dim=2)
            ll_attn2 = spat_attn2[i] * wtc_ll_in_levels[i].unsqueeze(axis=2)
            hl_attn2 = attn_hist2[i] * wtc_h_in_levels[i][:, :, 0:1, :, :]
            lh_attn2 = attn_hist2[i] * wtc_h_in_levels[i][:, :, 1:2, :, :]
            hh_attn2 = attn_hist2[i] * wtc_h_in_levels[i][:, :, 2:3, :, :]
            h_attn2 = torch.cat([hl_attn2, lh_attn2, hh_attn2], dim=2)
            # ======================================================
            ll_x1.append(ll_attn)
            h_x1.append(h_attn)
            ll_x2.append(ll_attn2)
            h_x2.append(h_attn2)
        return ll_x1, h_x1, ll_x2, h_x2

    def forward(self, x1, x2): 
        # ======================================================
        wt_in_levels, wt_ll_in_levels, wt_h_in_levels, shapes_in_levels = self.Multi_WTs(x1)
        wtc_ll_in_levels, wtc_h_in_levels = self.WTinConv(wt_in_levels)
        output_guass = self.guass_modeling(wt_ll_in_levels)
        _, attn_hist1 = self.hist_modeling(wt_h_in_levels)
        spat_attn1 = self.spat_attn(output_guass)
        # ======================================================
        wt_in_levels2, wt_ll_in_levels2, wt_h_in_levels2, shapes_in_levels2 = self.Multi_WTs(x2)
        wtc_ll_in_levels2, wtc_h_in_levels2 = self.WTinConv(wt_in_levels2)
        output_guass2 = self.guass_modeling(wt_ll_in_levels2)
        _, attn_hist2 = self.hist_modeling(wt_h_in_levels2)
        spat_attn2 = self.spat_attn(output_guass2)
        # ======================================================
        ll_x1, h_x1, ll_x2, h_x2 = self.cross_attn(spat_attn1, attn_hist1, wtc_ll_in_levels, wtc_h_in_levels, 
                                                   spat_attn2, attn_hist2, wtc_ll_in_levels2, wtc_h_in_levels2)
        # ======================================================
        x1_tags = self.Multi_IWTs(ll_x1, h_x1, shapes_in_levels)
        x2_tags = self.Multi_IWTs(ll_x2, h_x2, shapes_in_levels2)
        # ======================================================
        return x1_tags[-1], x2_tags[-1]

