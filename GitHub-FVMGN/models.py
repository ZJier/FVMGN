import torch
import numpy as np
import torch.nn as nn
from img_encoder import IMG_Encoder
import torch.nn.functional as F
from collections import OrderedDict
from WTDis import WTDs
import random
import pywt


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
        
    def forward(self, emb_i, emb_j): 
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        # loss = (2 * self.batch_size) / torch.sum(loss_partial)
        return loss


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)), 
            ("gelu", QuickGELU()), 
            ("c_proj", nn.Linear(d_model * 4, d_model))
                    ])) # MLP
        self.ln_2 = LayerNorm(d_model)
        self.ln_3 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class LinPro(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim), 
            QuickGELU(), 
            nn.Linear(in_features=dim, out_features=out_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FVMGN(nn.Module):
    def __init__(self, embed_dim: int, inchannel, vision_patch_size: int, num_classes, # vision 
                 context_length: int, vocab_size: int, transformer_width: int, # text
                 transformer_heads: int, transformer_layers: int, datasetname):
        super().__init__()
        self.eps_err = 1e-6
        self.inchannel = inchannel
        self.datasetname = datasetname
        self.vocab_size = vocab_size
        self.context_length = context_length

        self.ln_final = LayerNorm(transformer_width)
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers,
                                       heads=transformer_heads, attn_mask=self.build_attention_mask())
        
        self.linpros1 = LinPro(embed_dim, embed_dim // 2)
        self.linpros2 = LinPro(embed_dim, embed_dim // 4)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.wtdis = WTDs(in_channels=self.inchannel // 2, kernel_size=3, wt_levels=1, wt_type='db1', imgsize=vision_patch_size)

        self.visual_hsi = IMG_Encoder(channels = self.inchannel // 2, num_classes = num_classes, 
                                      image_size = vision_patch_size, hidden_dim = embed_dim // 8, emb_dim = embed_dim // 2)
        self.visual_other = IMG_Encoder(channels = self.inchannel // 2, num_classes = num_classes, 
                                        image_size = vision_patch_size, hidden_dim = embed_dim // 8, emb_dim = embed_dim // 2)

        self.conloss = ContrastiveLoss(batch_size=128)

        self.initialize_parameters()

    def initialize_parameters(self): 
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks: 
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual_hsi.conv1.weight.dtype

    def chan_div(self, x, mode): 
        hsi_data = x[:, :self.inchannel//2, :, :]
        other_data = x[:, self.inchannel//2:, :, :]
        return hsi_data, other_data
    
    def encode_hsi(self, hsi_img, mode): 
        return self.visual_hsi(hsi_img)
    
    def encode_other(self, other_img, mode): 
        return self.visual_other(other_img)

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection 
        return x
    
    def FeaAlign(self, inp_img, inp_txt, scale_factor, label_): 
        logits_per_img = scale_factor * inp_img @ inp_txt.t()
        logits_per_txt = scale_factor * inp_txt @ inp_img.t()
        loss_img = F.cross_entropy(logits_per_img, label_.long())
        loss_txt = F.cross_entropy(logits_per_txt, label_.long())
        loss = (loss_img + loss_txt) / 2
        return loss
    
    def MSPro_txt(self, txt, txt1, txt2): 
        text_fea3 = self.encode_text(txt)
        text_fea2 = self.linpros1(text_fea3)
        text_fea1 = self.linpros2(text_fea3)
        text_fea3_q1 = self.encode_text(txt1)
        text_fea2_q1 = self.linpros1(text_fea3_q1)
        text_fea1_q1 = self.linpros2(text_fea3_q1)
        text_fea3_q2 = self.encode_text(txt2)
        text_fea2_q2 = self.linpros1(text_fea3_q2)
        text_fea1_q2 = self.linpros2(text_fea3_q2)
        return text_fea1, text_fea2, text_fea3, text_fea1_q1, text_fea2_q1, text_fea3_q1, text_fea1_q2, text_fea2_q2, text_fea3_q2
    
    def Norm_fea(self, inps1, inps2, inps3): 
        inps1 = inps1 / inps1.norm(dim=1, keepdim=True)
        inps2 = inps2 / inps2.norm(dim=1, keepdim=True)
        inps3 = inps3 / inps3.norm(dim=1, keepdim=True)
        return inps1, inps2, inps3
    
    def WTs(self, inps): 
        LL_list = []
        LH_list = []
        for row in inps: 
            coeffs = pywt.wavedec(row.cpu().detach().numpy(), wavelet='haar', level=1)
            LL, LH = coeffs[0], coeffs[1]
            LL_list.append(LL), LH_list.append(LH)
        return torch.tensor(np.array(LL_list)).cuda(), torch.tensor(np.array(LH_list)).cuda()
    
    def forward(self, image, text, label, text_queue_1=None, text_queue_2=None): 
        # ======================================================
        hsi_data, other_data = self.chan_div(image, mode='train')
        # ======================================================
        hsi_data, other_data = self.wtdis(hsi_data, other_data)
        # ======================================================
        img_prob_hsi, img_fea_hsi1, img_fea_hsi2, img_fea_hsi3 = self.encode_hsi(hsi_data, mode='train')
        img_prob_other, img_fea_other1, img_fea_other2, img_fea_other3 = self.encode_other(other_data, mode='train')
        img_prob = torch.cat([img_prob_hsi, img_prob_other], axis=1)
        hsi_fea, oth_fea = img_fea_hsi3, img_fea_other3
        # ======================================================
        if self.training: 
            # ======================================================
            text_fea1, text_fea2, text_fea3, \
                text_fea1_q1, text_fea2_q1, text_fea3_q1, \
                    text_fea1_q2, text_fea2_q2, text_fea3_q2 = self.MSPro_txt(text, text_queue_1, text_queue_2)
            # ====================================================
            # Norm Features
            text_fea1, text_fea2, text_fea3 = self.Norm_fea(text_fea1, text_fea2, text_fea3)
            text_fea1_q1, text_fea2_q1, text_fea3_q1 = self.Norm_fea(text_fea1_q1, text_fea2_q1, text_fea3_q1)
            text_fea1_q2, text_fea2_q2, text_fea3_q2 = self.Norm_fea(text_fea1_q2, text_fea2_q2, text_fea3_q2)
            img_fea_hsi1, img_fea_hsi2, img_fea_hsi3 = self.Norm_fea(img_fea_hsi1, img_fea_hsi2, img_fea_hsi3)
            img_fea_other1, img_fea_other2, img_fea_other3 = self.Norm_fea(img_fea_other1, img_fea_other2, img_fea_other3)
            # ====================================================
            logit_scale = self.logit_scale.exp()
            loss1_q1 = self.FeaAlign(img_fea_hsi1, text_fea1_q1, logit_scale, label)
            loss2_q1 = self.FeaAlign(img_fea_hsi2, text_fea2_q1, logit_scale, label)
            loss3_q1 = self.FeaAlign(img_fea_hsi3, text_fea3_q1, logit_scale, label)
            loss1_q2 = self.FeaAlign(img_fea_other1, text_fea1_q2, logit_scale, label)
            loss2_q2 = self.FeaAlign(img_fea_other2, text_fea2_q2, logit_scale, label)
            loss3_q2 = self.FeaAlign(img_fea_other3, text_fea3_q2, logit_scale, label)
            img_fusion1 = (img_fea_hsi1 + img_fea_other1) / 2
            img_fusion2 = (img_fea_hsi2 + img_fea_other2) / 2
            img_fusion3 = (img_fea_hsi3 + img_fea_other3) / 2
            loss_clip1 = self.FeaAlign(img_fusion1, text_fea1, logit_scale, label)
            loss_clip2 = self.FeaAlign(img_fusion2, text_fea2, logit_scale, label)
            loss_clip3 = self.FeaAlign(img_fusion3, text_fea3, logit_scale, label)
            loss_vision1 = self.conloss(img_fea_hsi1, img_fea_other1)
            loss_vision2 = self.conloss(img_fea_hsi2, img_fea_other2)
            loss_vision3 = self.conloss(img_fea_hsi3, img_fea_other3)
            loss_ms1 = (loss_vision1 + loss_clip1 + loss1_q1 + loss1_q2) / 4
            loss_ms2 = (loss_vision2 + loss_clip2 + loss2_q1 + loss2_q2) / 4
            loss_ms3 = (loss_vision3 + loss_clip3 + loss3_q1 + loss3_q2) / 4
            # ====================================================
            loss_time = (loss_ms1 + loss_ms2 + loss_ms3) / 3
            # ====================================================
            text_wt1_ll, text_wt1_lh = self.WTs(text_fea1)
            text_wt2_ll, text_wt2_lh = self.WTs(text_fea2)
            text_wt3_ll, text_wt3_lh = self.WTs(text_fea3)
            text_wt1_q1_ll, text_wt1_q1_lh = self.WTs(text_fea1_q1)
            text_wt2_q1_ll, text_wt2_q1_lh = self.WTs(text_fea2_q1)
            text_wt3_q1_ll, text_wt3_q1_lh = self.WTs(text_fea3_q1)
            text_wt1_q2_ll, text_wt1_q2_lh = self.WTs(text_fea1_q2)
            text_wt2_q2_ll, text_wt2_q2_lh = self.WTs(text_fea2_q2)
            text_wt3_q2_ll, text_wt3_q2_lh = self.WTs(text_fea3_q2)
            img_wt_hsi1_ll, img_wt_hsi1_lh = self.WTs(img_fea_hsi1)
            img_wt_hsi2_ll, img_wt_hsi2_lh = self.WTs(img_fea_hsi2)
            img_wt_hsi3_ll, img_wt_hsi3_lh = self.WTs(img_fea_hsi3)
            img_wt_other1_ll, img_wt_other1_lh = self.WTs(img_fea_other1)
            img_wt_other2_ll, img_wt_other2_lh = self.WTs(img_fea_other2)
            img_wt_other3_ll, img_wt_other3_lh = self.WTs(img_fea_other3)
            # ====================================================
            loss1_q1_ll = self.FeaAlign(img_wt_hsi1_ll, text_wt1_q1_ll, logit_scale, label)
            loss1_q1_lh = self.FeaAlign(img_wt_hsi1_lh, text_wt1_q1_lh, logit_scale, label)
            loss2_q1_ll = self.FeaAlign(img_wt_hsi2_ll, text_wt2_q1_ll, logit_scale, label)
            loss2_q1_lh = self.FeaAlign(img_wt_hsi2_lh, text_wt2_q1_lh, logit_scale, label)
            loss3_q1_ll = self.FeaAlign(img_wt_hsi3_ll, text_wt3_q1_ll, logit_scale, label)
            loss3_q1_lh = self.FeaAlign(img_wt_hsi3_lh, text_wt3_q1_lh, logit_scale, label)
            # ====================================================
            loss1_q2_ll = self.FeaAlign(img_wt_other1_ll, text_wt1_q2_ll, logit_scale, label)
            loss1_q2_lh = self.FeaAlign(img_wt_other1_lh, text_wt1_q2_lh, logit_scale, label)
            loss2_q2_ll = self.FeaAlign(img_wt_other2_ll, text_wt2_q2_ll, logit_scale, label)
            loss2_q2_lh = self.FeaAlign(img_wt_other2_lh, text_wt2_q2_lh, logit_scale, label)
            loss3_q2_ll = self.FeaAlign(img_wt_other3_ll, text_wt3_q2_ll, logit_scale, label)
            loss3_q2_lh = self.FeaAlign(img_wt_other3_lh, text_wt3_q2_lh, logit_scale, label)
            # ====================================================
            img_wt1_ll = (img_wt_hsi1_ll + img_wt_other1_ll) / 2
            img_wt1_lh = (img_wt_hsi1_lh + img_wt_other1_lh) / 2
            img_wt2_ll = (img_wt_hsi2_ll + img_wt_other2_ll) / 2
            img_wt2_lh = (img_wt_hsi2_lh + img_wt_other2_lh) / 2
            img_wt3_ll = (img_wt_hsi3_ll + img_wt_other3_ll) / 2
            img_wt3_lh = (img_wt_hsi3_lh + img_wt_other3_lh) / 2
            # ====================================================
            loss1_sh_ll = self.FeaAlign(img_wt1_ll, text_wt1_ll, logit_scale, label)
            loss1_sh_lh = self.FeaAlign(img_wt1_lh, text_wt1_lh, logit_scale, label)
            loss2_sh_ll = self.FeaAlign(img_wt2_ll, text_wt2_ll, logit_scale, label)
            loss2_sh_lh = self.FeaAlign(img_wt2_lh, text_wt2_lh, logit_scale, label)
            loss3_sh_ll = self.FeaAlign(img_wt3_ll, text_wt3_ll, logit_scale, label)
            loss3_sh_lh = self.FeaAlign(img_wt3_lh, text_wt3_lh, logit_scale, label)
            # ====================================================
            loss_vis1_ll = self.conloss(img_wt_hsi1_ll, img_wt_other1_ll)
            loss_vis1_lh = self.conloss(img_wt_hsi1_lh, img_wt_other1_lh)
            loss_vis2_ll = self.conloss(img_wt_hsi2_ll, img_wt_other2_ll)
            loss_vis2_lh = self.conloss(img_wt_hsi2_lh, img_wt_other2_lh)
            loss_vis3_ll = self.conloss(img_wt_hsi3_ll, img_wt_other3_ll)
            loss_vis3_lh = self.conloss(img_wt_hsi3_lh, img_wt_other3_lh)
            # ====================================================
            # Divide Loss
            loss_wt_q1 = (loss1_q1_ll + loss1_q1_lh + loss2_q1_ll + loss2_q1_lh + loss3_q1_ll + loss3_q1_lh) / 6
            loss_wt_q2 = (loss1_q2_ll + loss1_q2_lh + loss2_q2_ll + loss2_q2_lh + loss3_q2_ll + loss3_q2_lh) / 6
            loss_wt_sh = (loss1_sh_ll + loss1_sh_lh + loss2_sh_ll + loss2_sh_lh + loss3_sh_ll + loss3_sh_lh) / 6
            loss_wt_vis = (loss_vis1_ll + loss_vis1_lh + loss_vis2_ll + loss_vis2_lh + loss_vis3_ll + loss_vis3_lh) / 6
            # ====================================================
            loss_feq = (loss_wt_q1 + loss_wt_q2 + loss_wt_sh + loss_wt_vis) / 4
            # ====================================================
            return loss_time, loss_feq, img_prob, hsi_fea, oth_fea
            # ====================================================
        else: 
            img_probs = torch.stack([img_prob_hsi, img_prob_other], dim=1)
            img_prob, _ = torch.max(img_probs, dim=1)
            return torch.tensor(0).long(), img_prob, hsi_fea, oth_fea
