import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Embeddings import (PatchEmbeddings, PositionalEmbeddings)
from Mixers import Transformer_ori, CTMixer, WCTMixer


class FRGCM(torch.nn.Module):
    def __init__(self, inp, outp=128, finp=30):
        super(FRGCM, self).__init__()
        self.inp = inp
        self.outp = outp
        self.outps = outp // 2
        self.finp = finp
        self.act = nn.ReLU(inplace=True)
        self.chan_trans = nn.Conv2d(inp, outp, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(self.outps)
        self.wtc_hsi_1 = nn.Conv2d(self.outps, self.outps, kernel_size=3, stride=1, padding=1, groups=self.outps)
        self.bn2 = nn.BatchNorm2d(self.outps)
        self.wtc_hsi_2 = nn.Conv2d(self.outps, self.outps, kernel_size=3, stride=1, padding=1, groups=self.outps)
        self.bn3 = nn.BatchNorm2d(self.outps)
        self.wtc_hsi_3 = nn.Conv2d(self.outps, self.outps, kernel_size=3, stride=1, padding=1, groups=self.outps)

        self.bno_1 = nn.BatchNorm2d(self.outps)
        self.wtc_img_1 = nn.Conv2d(self.outps, self.outps, kernel_size=3, stride=1, padding=1, groups=self.outps)
        self.bno_2 = nn.BatchNorm2d(self.outps)
        self.wtc_img_2 = nn.Conv2d(self.outps, self.outps, kernel_size=3, stride=1, padding=1, groups=self.outps)
        self.bno_3 = nn.BatchNorm2d(self.outps)
        self.wtc_img_3 = nn.Conv2d(self.outps, self.outps, kernel_size=3, stride=1, padding=1, groups=self.outps)

        self.bnf_1 = nn.BatchNorm2d(self.outps)
        self.wtc_fuse_1 = nn.Conv2d(self.outps, self.outps, kernel_size=3, stride=1, padding=1, groups=self.outps)
        self.bnf_2 = nn.BatchNorm2d(self.outps)
        self.wtc_fuse_2 = nn.Conv2d(self.outps, self.outps, kernel_size=3, stride=1, padding=1, groups=self.outps)

        self.bnf_3 = nn.BatchNorm2d(self.outps)
        self.wtc_fuse_3 = nn.Conv2d(self.outps, finp, kernel_size=1, stride=1, padding=0)

    def forward(self, x): 
        x_trans = self.chan_trans(x)

        x_trans_1 = x_trans[:, 0:self.outps, :, :]
        x_trans_2 = x_trans[:, self.outps:self.outp, :, :]

        fea1_hsi = self.wtc_hsi_1(self.act(self.bn1(x_trans_1)))
        fea2_hsi = self.wtc_hsi_2(self.act(self.bn2(fea1_hsi)))
        fea3_hsi = self.wtc_hsi_3(self.act(self.bn3(fea2_hsi)))
        fea1_img = self.wtc_img_1(self.act(self.bno_1(x_trans_2)))
        fea2_img = self.wtc_img_2(self.act(self.bno_2(fea1_img)))
        fea3_img = self.wtc_img_3(self.act(self.bno_3(fea2_img)))

        fea1_fuse = fea1_hsi + fea1_img
        fea1_fuse_fuse = self.wtc_fuse_1(self.act(self.bnf_1(fea1_fuse)))
        fea2_fuse = fea2_hsi + fea2_img + fea1_fuse_fuse
        fea2_fuse_fuse = self.wtc_fuse_2(self.act(self.bnf_2(fea2_fuse)))
        fea3_fuse = fea3_hsi + fea3_img + fea2_fuse_fuse
        out_fuse = self.wtc_fuse_3(self.act(self.bnf_3(fea3_fuse)))

        return out_fuse


class Pooling(nn.Module):
    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")
        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)


class Classifier(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(in_features=dim, out_features=num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LinPros(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.GELU(), 
            nn.Linear(in_features=dim, out_features=dim_out)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class IMG_Encoder(nn.Module): 
    def __init__(self, channels, num_classes, image_size, hidden_dim, emb_dim, patch_size = 1, pool: str = "mean"):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_patch = int(math.sqrt(self.num_patches))
        self.patch_dim = channels * patch_size ** 2

        self.attcross = FRGCM(inp=channels, outp=128, finp = channels)
        self.patch_embeddings = PatchEmbeddings(patch_size=self.patch_size, patch_dim=self.patch_dim, emb_dim=self.emb_dim)
        self.pos_embeddings = PositionalEmbeddings(num_pos=self.num_patches, dim=self.emb_dim)
        self.wct_branch = WCTMixer(dim=emb_dim, num_layers=1, num_heads=1, head_dim=self.hidden_dim)
        self.ct_branch = CTMixer(dim=emb_dim, num_layers=1, num_heads=4, head_dim=self.hidden_dim, 
                                         hidden_dim=self.hidden_dim, num_patch=self.num_patch, patch_size=patch_size)

        # self.dropout = nn.Dropout(0.5)
        self.pool = Pooling(pool=pool)
        self.classifier = Classifier(dim=emb_dim, num_classes=int(num_classes))
        self.classifier1 = LinPros(dim=emb_dim, dim_out=emb_dim // 2)
        self.classifier2 = LinPros(dim=emb_dim, dim_out=emb_dim)
        self.classifier3 = LinPros(dim=emb_dim, dim_out=emb_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attcross(x)
        x_patch = self.patch_embeddings(x)
        x_pos = self.pos_embeddings(x_patch)
        out_m = self.wct_branch(x_pos)
        out_c = self.ct_branch(x_pos)

        out_all = out_m + out_c
        x_fea = self.pool(out_all)

        x_cls = self.classifier(x_fea)
        x_fea1 = self.classifier1(x_fea)
        x_fea2 = self.classifier2(x_fea)
        x_fea3 = self.classifier3(x_fea)

        return x_cls, x_fea1, x_fea2, x_fea3