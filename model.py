# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
import scipy
import matplotlib.pyplot as plt
import numpy as np


# Relative Position Encoding ##
class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=32, max_distance=128, n_heads=4):
        super(RelativePositionBias, self).__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance):
        num_buckets //= 2
        ret = (relative_position < 0).to(relative_position) * num_buckets
        relative_position = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(relative_position / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1), )

        ret += torch.where(is_small, relative_position.long(), val_if_large)
        return ret.long()

    def forward(self, relative_position):
        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bias = self.relative_attention_bias(rp_bucket)
        return rp_bias

# End ##


# Ingredients of Transformer ##
class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, bias, loop):
        attn = torch.matmul(q / self.scale, k.transpose(-1, -2))
        if bias is not None:
            attn += bias
        if loop is not None:
            attn *= loop
        attn = F.softmax(attn, dim=-1)
        attn_m = self.dropout(attn)
        output = torch.matmul(attn_m, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        if d_model % n_head != 0:
            raise ValueError("The hidden size is not a multiple of the number of attention heads")

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.fc_query = nn.Linear(d_model, d_model, bias=False)
        self.fc_key = nn.Linear(d_model, d_model, bias=False)
        self.fc_value = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention(scale=self.d_k ** 0.5, dropout=dropout)
        self.fc_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """
        x has shape (*, L, C)
        return shape (*, nhead, L, C/nhead)
        """
        new_shape = x.shape[:-1] + (self.n_head, -1)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)

    def forward(self, x, bias, loop):
        q = self.transpose_for_scores(self.fc_query(x))
        k = self.transpose_for_scores(self.fc_key(x))
        v = self.transpose_for_scores(self.fc_value(x))
        x, attn = self.attention(q, k, v, bias, loop)
        x = x.transpose(-3, -2)
        x = x.reshape(*x.shape[:-2], -1)
        x = self.dropout(self.fc_out(x))
        return x, attn


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, x, bias, loop):
        branch, attn = self.attn(self.norm1(x), bias, loop)
        x = x + branch
        x = x + self.ffn(self.norm2(x))
        return x, attn


class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(**kwargs) for _ in range(n_layer)])

    def forward(self, x, bias, loop=None):
        attn_weight = []
        for module in self.layers:
            x, w = module(x, bias, loop)
            attn_weight.append(w)
        return x, attn_weight
# End ##


class WConv(nn.Module):
    def __init__(self, **kwargs):
        super(WConv, self).__init__()
        self.conv = nn.Conv1d(**kwargs)

    def forward(self, x):
        shape = x.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])
        x = x.transpose(-1, -2)
        x = self.conv(x)
        x = x.transpose(-1, -2)
        x = x.reshape(*shape, *x.shape[-2:])
        return x


class RefineT5(nn.Module):
    def __init__(self, fea_dim=4):
        super(RefineT5, self).__init__()
        # initilizing hper-parameters
        self.config()
        self.fea_dim = fea_dim
        self.seq_dim = 64
        self.d_model_1 = 256
        self.n_layer_1 = 2
        self.n_head_1 = self.d_model_1 // 64
        self.d_model_2 = 256
        self.n_layer_2 = 4
        self.n_head_2 = self.d_model_2 // 64
        #
        self.embed_seq = nn.Embedding(5, self.seq_dim, padding_idx=-1)
        self.projection = nn.Sequential(
            WConv(
                in_channels=self.seq_dim,
                out_channels=self.d_model_1,
                kernel_size=1,
                stride=1,
                padding=0), 
            nn.Dropout(0.1)
        )
        
        self.embed_signal = nn.Sequential(
            nn.Linear(1, self.d_model_1),
            nn.Dropout(0.1)
        )
        #
        self.rp_bias_1 = RelativePositionBias(
            num_buckets=64, max_distance=256, n_heads=self.n_head_1
        )
        self.encoder_mask = TransformerEncoder(
            n_layer=2,
            d_model=256,
            n_head=4,
            dim_feedforward=256 * 4,
            dropout=0.1,
        )

        self.mask_output = nn.Linear(self.d_model_1, 2)
        #
        self.encoder_stage1 = TransformerEncoder(
            n_layer=self.n_layer_1,
            d_model=self.d_model_1,
            n_head=self.n_head_1,
            dim_feedforward=self.d_model_1 * 4,
            dropout=0.1,
        ) 
        #
        if self.loop_in:
            self.embed_multi_signal = nn.Sequential(
                nn.Linear(self.fea_dim+1, self.d_model_2),
                nn.Dropout(0.1)
            )
        else:
            self.embed_multi_signal = nn.Sequential(
                nn.Linear(self.fea_dim, self.d_model_2),
                nn.Dropout(0.1)
            )
        self.rp_bias_2 = RelativePositionBias(
            num_buckets=16, max_distance=64, n_heads=self.n_head_2
        )
        self.encoder_stage2 = TransformerEncoder(
            n_layer=self.n_layer_2,
            d_model=self.d_model_2,
            n_head=self.n_head_2,
            dim_feedforward=self.d_model_2 * 4,
            dropout=0.1,
        )
        self.layer_norm = nn.LayerNorm(self.d_model_2)
        self.linear_out = nn.Sequential(
                    nn.Linear(self.d_model_2 + self.rna_feat_dim if self.useRNAFeat else self.d_model_2, 64),
                    nn.GELU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(64, 1)
                    )
        self.activation = nn.Softplus()

    def config(self):
        # parameters of beta distribution 
        self.beta_min = 1
        self.prior_scale_factor = 10
        self.signal_beta = 2
        self.z_scale = 1
        self.only_sig = True # default: True
        self.training = True
        self.aux_loss_kl = True
        self.mult_signal_beta = 2
        self.exponent = True
        self.useRNAFeat = True
        self.rna_feat_dim = 8
        # marginal distribution p(z)
        self.marginal_mean = 0.1
        self.marginal_alpha = 1.0
        self.marginal_beta = (1 - self.marginal_mean) / self.marginal_mean * self.marginal_alpha
        # prior distribution
        self.prior_mean = 0.8 
        self.prior_alpha_pos = 8.0 
        self.prior_beta_pos = (1 - self.prior_mean) / self.prior_mean * self.prior_alpha_pos
        self.hidden = True
        self.rp = False
        self.loop_in = False
        self.loop_w = False
        # 
        self.marginal_mean2 = 0.8 
        self.marginal_alpha2 = 8.0 
        self.marginal_beta2 = (1 - self.marginal_mean2) / self.marginal_mean2 * self.marginal_alpha2

    def set_train(self):
        self.training = True

    def set_eval(self):
        self.training = False

    def _relative_position_embed(self, pos_rp):
        rp_matrix = pos_rp.unsqueeze(-1) - pos_rp.unsqueeze(-2)
        rp_bias = self.rp_bias_1(rp_matrix)
        rp_bias = rp_bias.permute(2, 0, 1)

        return rp_bias
    
    def _relative_position(self, pos_rp):
        rp_matrix = pos_rp.unsqueeze(-1) - pos_rp.unsqueeze(-2)
        rp_bias = self.rp_bias_2(rp_matrix)
        rp_bias = rp_bias.permute(2, 0, 1)

        return rp_bias
    
    def _relative_distance(self, position):
        row, _ = position.size()
        TSS = position[:, 0].view(row, -1)
        pos_rd = torch.div(torch.abs(position - TSS), 1000, rounding_mode='floor')
        rd_matrix = pos_rd.unsqueeze(-1) - pos_rd.unsqueeze(-2)
        rd_bias = self.rp_bias_2(rd_matrix)
        rd_bias = rd_bias.permute(0, 3, 1, 2)

        return rd_bias

    def posterior_dist_logit(self, logit):
        alpha_logits = logit[..., 1] 
        beta_logits = logit[..., 0] 

        alpha = (F.softplus(alpha_logits) + self.beta_min) * self.prior_scale_factor
        beta = (F.softplus(beta_logits) + self.beta_min)  * self.prior_scale_factor

        self.post_distribution_seq = dist.Beta(alpha, beta)
    
    def posterior_dist_signal(self, signal):
        prior_signal = torch.exp(signal)
        cur_alpha = prior_signal * self.prior_scale_factor 
        cur_beta = self.signal_beta * self.prior_scale_factor   
        self.post_distribution_sig = dist.Beta(cur_alpha, cur_beta)
    
    def couple_post_dist(self):
        seq_alpha, seq_beta = self.post_distribution_seq.concentration1, self.post_distribution_seq.concentration0
        sig_alpha, sig_beta = self.post_distribution_sig.concentration1, self.post_distribution_sig.concentration0
        # merge two distribution
        x_alpha = seq_alpha + sig_alpha 
        x_beta = seq_beta + sig_beta
        x_alpha = x_alpha * self.z_scale
        x_beta = x_beta * self.z_scale
        self.z_distribution = dist.Beta(x_alpha, x_beta)

    def prior_dist_stage1(self):
        marginal_alpha = self.marginal_alpha * self.prior_scale_factor
        marginal_beta = self.marginal_beta * self.prior_scale_factor
        self.prior_distribution_1 = dist.Beta(marginal_alpha, marginal_beta)

    def posterior_dist_stage2(self, multi_signals):
        multi_signals = torch.exp(multi_signals)
        signal_dim = multi_signals.shape[-1]
        geometric_mean = 1
        for i in range(signal_dim):
            geometric_mean *= multi_signals[..., i]
        geometric_mean = torch.pow(geometric_mean, 1/signal_dim)
        # alpha = (geometric_mean + self.beta_min) * self.prior_scale_factor
        alpha = geometric_mean * self.prior_scale_factor
        beta = self.mult_signal_beta * self.prior_scale_factor
        self.post_distribution_multi_sig = dist.Beta(alpha, beta)
    
        
    def prior_dist_stage2(self, position, decay_rate=0.001):
        row, _ = position.size()
        TSS = position[:, 0].view(row, -1)
        distance = torch.div(torch.abs(position - TSS), 1000, rounding_mode='floor')
        if self.exponent:
            beta_add = (1-torch.exp(-decay_rate*distance))*distance
            beta_add = beta_add.long()
        else:
            max_, _ = torch.max(distance, dim=1, keepdim=True)
            beta_add = distance / max_ * distance
            beta_add = beta_add.long()
        
        alpha = self.prior_alpha_pos * self.prior_scale_factor
        beta = (self.prior_beta_pos + beta_add) * self.prior_scale_factor
        self.prior_distribution_2 = dist.Beta(alpha, beta)
        
    def kl_divergence_stage1(self):
        if self.only_sig:
            post_dist = self.post_distribution_sig
            prior_dist = self.prior_distribution_1
            kl_loss = dist.kl_divergence(post_dist, prior_dist)
            kl_loss = torch.mean(torch.mean(kl_loss, dim=-1), dim=-1)
            return kl_loss

        post_dist = self.z_distribution
        prior_dist = self.prior_distribution_1
        kl_loss = dist.kl_divergence(post_dist, prior_dist)
        kl_loss = torch.mean(torch.mean(kl_loss, dim=-1), dim=-1)
        return kl_loss

    def kl_divergence_stage2(self):
        post_dist = self.post_distribution_multi_sig
        prior_dist = self.prior_distribution_2
        kl_loss = dist.kl_divergence(post_dist, prior_dist)
        kl_loss = torch.mean(kl_loss, dim=-1)
        return kl_loss

    def aux_loss(self):
        kl_loss_1 = self.kl_divergence_stage1()
        kl_loss_2 = self.kl_divergence_stage2()
        aux_loss = {
            'kl_loss_1': kl_loss_1,
            'kl_loss_2': kl_loss_2,
        }
        return aux_loss

    def forward(self, seqs, signal, position, loop_input, loop_weight):
        """
        :param seq: B*M*L, M means the number of cCREs and L means the sequence length
        :param signal: dict{B*M*L, B*M*D}
        :param distance: M*M
        :return: N*1
        """
        seq, seq_rc = seqs['seq'], seqs['seq_rc']
        _, M, L = seq.size()
        seq_embed = self.embed_seq(seq)
        seq_embed = self.projection(seq_embed)
        seq_embed_rc = self.embed_seq(seq_rc)
        seq_embed_rc = self.projection(seq_embed_rc)
        seq_embed += seq_embed_rc
        #
        pos_rp_1 = torch.arange(L, device=seq.device)
        rp_bias_1 = self._relative_position_embed(pos_rp_1)
        # calculate the mask distribution
        chromatin_signal = signal['DNase']
        sig_embed = self.embed_signal(chromatin_signal.unsqueeze(-1))
        self.posterior_dist_signal(chromatin_signal)
        self.prior_dist_stage1()
        if self.only_sig:
            z_dist_1 = self.post_distribution_sig
        else:
            seq_encode, _ = self.encoder_mask(seq_embed, rp_bias_1)
            logits = self.mask_output(seq_encode)
            self.posterior_dist_logit(logits)
            self.couple_post_dist()
            z_dist_1 = self.z_distribution
        #
        if not self.training:
            soft_mask = z_dist_1.mean
            mask = soft_mask
        else:
            soft_mask = z_dist_1.rsample()
            mask = soft_mask.clone()
        # retain promoter regions
        mask[:, 0, :] = 1.0
        input_embed_enc = seq_embed + sig_embed       
        input_embed_enc = input_embed_enc * mask.unsqueeze(-1)
        hidden_states, _ = self.encoder_stage1(input_embed_enc, rp_bias_1)
        #
        hidden_states = torch.mean(hidden_states, dim=-2)
        # the second stage
        multi_signals = signal['multi-signal']
        if self.loop_in:
            loop_in = torch.unsqueeze(loop_input[:, :, 0], -1)
            multi_signals = torch.cat((multi_signals, loop_in), dim=-1)
        multi_signal_embed = self.embed_multi_signal(multi_signals)
        self.posterior_dist_stage2(multi_signals)
        self.prior_dist_stage2(position)
        z_dist_2 = self.post_distribution_multi_sig
        if not self.training:
            soft_mask = z_dist_2.mean
            mask = soft_mask
        else:
            soft_mask = z_dist_2.rsample()
            mask = soft_mask.clone()
        # retain promoter regions
        mask[:, 0] = 1.0
        aux_infor = self.aux_loss()
        aux_infor['mask_cCRE'] = mask
        ccre_embed_enc = hidden_states + multi_signal_embed
        ccre_embed_enc = ccre_embed_enc * mask.unsqueeze(-1)
        # relative position; is msk_bias needed?
        if self.rp:
            pos_rp_2 = torch.arange(M, device=seq.device)
            rp_bias_2 = self._relative_position(pos_rp_2)
        else:
            rp_bias_2 = self._relative_distance(position)
        if self.hidden:
            attn_mask = (torch.eye(M, device=seq.device)>0)
            msk_bias = torch.zeros((M, M), device=seq.device)
            msk_bias = msk_bias.masked_fill(attn_mask, float('-inf'))
        else:
            msk_bias = torch.zeros((M, M), device=seq.device)
        if self.loop_w:
            loop_w = torch.unsqueeze(loop_weight, dim=1)
            loop_w = torch.exp(loop_w)            
        output_embed, attn_weight = self.encoder_stage2(ccre_embed_enc, rp_bias_2+msk_bias, loop_w)
        # extract the gene row
        output_embed = self.layer_norm(output_embed)
        gene_embed = output_embed[:, 0, :]
        if self.useRNAFeat:
            rna_feat = signal['rna_feat']
            gene_embed = torch.cat([gene_embed, rna_feat], dim=-1)
        y = self.linear_out(gene_embed)
        y = self.activation(y)

        return y, aux_infor, attn_weight


class TransformerT5(nn.Module):
    def __init__(self, fea_dim=4):
        super(TransformerT5, self).__init__()
        # initilizing hper-parameters
        self.config()
        self.fea_dim = fea_dim
        self.seq_dim = 64
        self.d_model_1 = 256
        self.n_layer_1 = 2
        self.n_head_1 = self.d_model_1 // 64
        self.d_model_2 = 256
        self.n_layer_2 = 4
        self.n_head_2 = self.d_model_2 // 64
        #
        self.embed_seq = nn.Embedding(5, self.seq_dim, padding_idx=-1)
        self.projection = nn.Sequential(
            WConv(
                in_channels=self.seq_dim,
                out_channels=self.d_model_1,
                kernel_size=1,
                stride=1,
                padding=0), 
            nn.Dropout(0.1)
        )
        
        self.embed_signal = nn.Sequential(
            nn.Linear(1, self.d_model_1),
            nn.Dropout(0.1)
        )
        #
        self.rp_bias_1 = RelativePositionBias(
            num_buckets=64, max_distance=256, n_heads=self.n_head_1
        )
        #
        self.encoder_stage1 = TransformerEncoder(
            n_layer=self.n_layer_1,
            d_model=self.d_model_1,
            n_head=self.n_head_1,
            dim_feedforward=self.d_model_1 * 4,
            dropout=0.1,
        ) 
        #
        if self.loop_in:
            self.embed_multi_signal = nn.Sequential(
                nn.Linear(self.fea_dim+1, self.d_model_2),
                nn.Dropout(0.1)
            )
        else:
            self.embed_multi_signal = nn.Sequential(
                nn.Linear(self.fea_dim, self.d_model_2),
                nn.Dropout(0.1)
            )
        self.rp_bias_2 = RelativePositionBias(
            num_buckets=16, max_distance=64, n_heads=self.n_head_2
        )
        self.encoder_stage2 = TransformerEncoder(
            n_layer=self.n_layer_2,
            d_model=self.d_model_2,
            n_head=self.n_head_2,
            dim_feedforward=self.d_model_2 * 4,
            dropout=0.1,
        )
        self.layer_norm = nn.LayerNorm(self.d_model_2)
        self.linear_out = nn.Sequential(
                    nn.Linear(self.d_model_2 + self.rna_feat_dim if self.useRNAFeat else self.d_model_2, 64),
                    nn.GELU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(64, 1)
                    )
        self.activation = nn.Softplus()

    def config(self):
        self.training = True
        self.useRNAFeat = True
        self.rna_feat_dim = 8
        self.hidden = False
        self.rp = False
        self.loop_in = False
        self.loop_w = False

    def set_train(self):
        self.training = True

    def set_eval(self):
        self.training = False

    def _relative_position_1(self, pos_rp):
        rp_matrix = pos_rp.unsqueeze(-1) - pos_rp.unsqueeze(-2)
        rp_bias = self.rp_bias_1(rp_matrix)
        rp_bias = rp_bias.permute(2, 0, 1)

        return rp_bias
    
    def _relative_position(self, pos_rp):
        rp_matrix = pos_rp.unsqueeze(-1) - pos_rp.unsqueeze(-2)
        rp_bias = self.rp_bias_2(rp_matrix)
        rp_bias = rp_bias.permute(2, 0, 1)

        return rp_bias
    
    def _relative_distance(self, position):
        row, _ = position.size()
        TSS = position[:, 0].view(row, -1)
        pos_rd = torch.div(torch.abs(position - TSS), 1000, rounding_mode='floor')
        rd_matrix = pos_rd.unsqueeze(-1) - pos_rd.unsqueeze(-2)
        rd_bias = self.rp_bias_2(rd_matrix)
        rd_bias = rd_bias.permute(0, 3, 1, 2)

        return rd_bias

    def forward(self, seqs, signal, position, loop_input, loop_weight):
        """
        :param seq: B*M*L, M means the number of cCREs and L means the sequence length
        :param signal: dict{B*M*L, B*M*D}
        :param distance: M*M
        :return: N*1
        """
        seq, seq_rc = seqs['seq'], seqs['seq_rc']
        _, M, L = seq.size()
        seq_embed = self.embed_seq(seq)
        seq_embed = self.projection(seq_embed)
        #
        seq_embed_rc = self.embed_seq(seq_rc)
        seq_embed_rc = self.projection(seq_embed_rc)
        #
        seq_embed += seq_embed_rc
        pos_rp_1 = torch.arange(L, device=seq.device)
        rp_bias_1 = self._relative_position_1(pos_rp_1)
        chromatin_signal = signal['DNase']
        sig_embed = self.embed_signal(chromatin_signal.unsqueeze(-1))
        input_embed_enc = seq_embed + sig_embed
        hidden_states, _ = self.encoder_stage1(input_embed_enc, rp_bias_1)
        #
        hidden_states = torch.mean(hidden_states, dim=-2)
        # the second stage
        multi_signals = signal['multi-signal']
        if self.loop_in:
            loop_in = torch.unsqueeze(loop_input[:, :, 0], -1)
            multi_signals = torch.cat((multi_signals, loop_in), dim=-1)
        multi_signal_embed = self.embed_multi_signal(multi_signals)
        ccre_embed_enc = hidden_states + multi_signal_embed
        # relative position; is msk_bias needed?
        if self.rp:
            pos_rp_2 = torch.arange(M, device=seq.device)
            rp_bias_2 = self._relative_position(pos_rp_2)
        else:
            rp_bias_2 = self._relative_distance(position)
        if self.hidden:
            attn_mask = (torch.eye(M, device=seq.device)>0)
            msk_bias = torch.zeros((M, M), device=seq.device)
            msk_bias = msk_bias.masked_fill(attn_mask, float('-inf'))
        else:
            msk_bias = torch.zeros((M, M), device=seq.device)
        loop_w = None
        if self.loop_w:
            loop_w = torch.unsqueeze(loop_weight, dim=1)
            loop_w = torch.exp(loop_w)            
        output_embed, attn_weight = self.encoder_stage2(ccre_embed_enc, rp_bias_2+msk_bias, loop_w)
        # extract the gene row
        output_embed = self.layer_norm(output_embed)
        gene_embed = output_embed[:, 0, :]
        if self.useRNAFeat:
            rna_feat = signal['rna_feat']
            gene_embed = torch.cat([gene_embed, rna_feat], dim=-1)
        y = self.linear_out(gene_embed)
        y = self.activation(y)

        return y, attn_weight