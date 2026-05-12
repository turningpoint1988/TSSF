#!/usr/bin/python

import os
from copy import copy
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils import data
import torch.nn.functional as F

import h5py
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def scatterplot(pred, true, out_f):
    df = pd.DataFrame({'Prediction': pred, 'True': true})
    max_p, max_t = int(np.ceil(np.max(pred))), int(np.ceil(np.max(true)))
    x = range(0, np.maximum(max_p, max_t))
    sns.set_theme(style="dark")
    fig, ax = plt.subplots()
    sns.despine(fig)
    sns.scatterplot(data=df, x="Prediction", y="True", s=10, ax=ax)
    ax.plot(x, x, '--', linewidth=0.8, color='grey')
    plt.savefig(out_f, format='png', bbox_inches='tight', dpi=300)


class SourceDataSet(data.Dataset):
    def __init__(self, seq, seq_rc, dnase, multi_signal, position, expression, rna, loop1=None, loop2=None):
        super(SourceDataSet, self).__init__()
        self.seq = seq
        self.seq_rc = seq_rc
        self.dnase = dnase
        self.signal = multi_signal
        self.position = position
        self.expr = expression
        self.rna = rna
        self.loop1 = loop1
        self.loop2 = loop2

        assert len(self.seq) == len(self.expr) and len(self.signal) == len(self.expr), \
        "the number of sequences and labels must be consistent."

    def __len__(self):
        return len(self.expr)

    def __getitem__(self, index):
        seq_one = self.seq[index]
        seq_rc_one = self.seq_rc[index]
        dnase_one = self.dnase[index]
        signal_one = self.signal[index]
        position_one = self.position[index]
        expr_one = self.expr[index]
        rna_one = self.rna[index]
        if self.loop1 is None and self.loop2 is None:
            return {"seq": seq_one, "seq_rc": seq_rc_one, "dnase": dnase_one, 'multi-signal': signal_one, 
                    "position": position_one, "expr": expr_one, "rna": rna_one}
        else:
            loop1_one = self.loop1[index]
            loop2_one = self.loop2[index]
            return {"seq": seq_one, "seq_rc": seq_rc_one, "dnase": dnase_one, 'multi-signal': signal_one, 
                    "position": position_one, "expr": expr_one, "rna": rna_one, "loop1": loop1_one, "loop2": loop2_one}


class Lossmetrics(nn.Module):
    def __init__(self, weight=0.01):
        super(Lossmetrics, self).__init__()
        self.kl_loss_weight = weight

    def mse(self, prediction, target):
        prediction = prediction.view(-1)
        target = target.view(-1)
        loss = F.mse_loss(prediction, target)
        return loss
    
    def l1_loss(self, prediction, target):
        prediction = prediction.view(-1)
        target = target.view(-1)
        loss = F.smooth_l1_loss(prediction, target)
        return loss

    def fusion(self, prediction, target, aux_infor):
        prediction = prediction.view(-1)
        target = target.view(-1)
        loss = F.smooth_l1_loss(prediction, target) + \
            self.kl_loss_weight*torch.mean(aux_infor['kl_loss_1']) + \
            self.kl_loss_weight*torch.mean(aux_infor['kl_loss_2'])
        return loss
    
    def normalization(self, x):
        x_min, _ = torch.min(x, dim=-1, keepdim=True)
        x_max, _ = torch.max(x, dim=-1, keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min)
        return x_norm

    def l1_loss_kl(self, pred, target, att_weights, position):
        pred = pred.view(-1)
        target = target.view(-1)
        loss = F.smooth_l1_loss(pred, target)
        # 
        attn_integrated = torch.zeros_like(att_weights[0])
        for att in att_weights:
            attn_integrated += att
        attn_integrated = torch.mean(attn_integrated, dim=1)
        N, M, _ = attn_integrated.size()
        att_gene = attn_integrated[:, 0, :]
        att_gene = self.normalization(att_gene)
        TSS = position[:, 0].view(N, 1)
        val = torch.abs(position-TSS) / 1000
        k = 1.0
        decay = 2 / (1 + torch.exp(k*val))
        decay = self.normalization(decay)
        # 
        index = (target > 0.0)
        if torch.sum(index) == 0:
            return loss
        else:
            att_gene = att_gene[index]
            decay = decay[index]
            loss_decay = F.kl_div(att_gene, decay, reduction='mean')
            loss += self.decay_loss_weight * loss_decay
            return loss


class Trainer(object):
    """build a trainer"""
    def __init__(self, model, data_dir, checkpoint):
        self.config()
        self.model = model.to(self.device)
        self.data_dir = data_dir
        self.checkpoint = checkpoint
        self.optimizer = optim.AdamW(self.model.parameters(), betas=self.betas, lr=self.lr_fixed, weight_decay=self.weight_decay)
        self.criterion = Lossmetrics(self.kl_loss_weight)
    
    def config(self):
        self.species = 'Human' # Mouse
        self.batch_size = 8 # cCRE=10, batch_size=32; cCRE=50, batch_size=8
        self.betas = (0.9, 0.95)
        # params for training network
        self.lr_fixed = 0.0005
        self.lr_up = 1e-03 
        self.lr_low = 1e-06
        self.weight_decay = 1e-06
        self.warmup_step = 6000 
        self.train_step = 24000 
        self.eval_step = 2000
        self.num_epochs = 200
        self.kl_loss_weight = 0.01 
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.gradient_clip = 1
        # 
        self.tr_count = 6 # cCRE=10, tr_count=1; cCRE=50, tr_count=6
        self.te_count = 1
        self.va_count = 1
        self.loss_best = float('inf')
        self.state_dict = None
        self.step_size = 100
        self.loop = True
    
    def _increase_lr(self, step):
        lr_fraction = np.minimum(1.0, step / np.maximum(1, self.train_step))
        lr_current = self.lr_low + (self.lr_up - self.lr_low) * lr_fraction
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_current

    def train(self, warmup_state=False):
        """training the model"""
        self.model.train()
        if torch.cuda.device_count() > 1:
            self.model.module.set_train()
        else:
            self.model.set_train()
        train_step = self.warmup_step if warmup_state else self.train_step
        index = 0
        step = 0
        for _ in range(self.num_epochs):
            for count in range(self.tr_count):
                with h5py.File(self.data_dir + '/tr_{}.hdf5'.format(count + 1), 'r') as f:
                    sequence = np.array(f['seq'])
                    sequence_rc = np.array(f['seq_rc'])
                    dnase = np.array(f['DNase'])
                    signal = np.array(f['signal'])
                    position = np.array(f['position'])
                    expression = np.array(f['expr'])
                    rna_feat = np.array(f['rna'])
                    if self.loop:
                        loop_input = np.array(f['loop1'])
                        loop_weight = np.array(f['loop2'])
                    else:
                        loop_input = None
                        loop_weight = None
                tr_loader = DataLoader(SourceDataSet(sequence, sequence_rc, dnase, signal, position, expression, rna_feat, loop_input, loop_weight), 
                                       batch_size=self.batch_size, shuffle=True)
                for _, sample_batch in enumerate(tr_loader):
                    if step <= self.train_step and step % self.step_size == 0:
                        self._increase_lr(step)
                    seq = sample_batch["seq"].long().to(self.device)
                    seq_rc = sample_batch["seq_rc"].long().to(self.device)
                    dnase = sample_batch["dnase"].float().to(self.device)
                    multi_signal = sample_batch["multi-signal"].float().to(self.device)
                    position = sample_batch["position"].float().to(self.device)
                    expr = sample_batch["expr"].float().to(self.device)
                    rna_feat = sample_batch["rna"].float().to(self.device)
                    if self.loop:
                        loop_in = sample_batch["loop1"].float().to(self.device)
                        loop_w = sample_batch["loop2"].float().to(self.device)
                    else:
                        loop_in = None
                        loop_w = None
                    seqs = {'seq': seq, 'seq_rc': seq_rc}
                    signals = {'DNase': dnase, 'multi-signal': multi_signal, 'rna_feat': rna_feat}
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    preds = self.model(seqs, signals, position, loop_in, loop_w)
                    loss = self.criterion.fusion(preds[0], expr, preds[1]) # the loss for RefineT5
                    # loss = self.criterion.l1_loss(preds[0], expr) # the loss for TransformerT5
                    # loss = self.criterion.l1_loss_kl(preds[0], expr, preds[-1], position) # the loss for TransformerT5+KL
                    if np.isnan(loss.item()):
                        raise ValueError('loss is nan while training')
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()
                    # validation and save the model with higher accuracy
                    if step % self.eval_step == 0:
                        loss_val = self.validation()
                        print("step/{}--loss_tr/{:.3f}--loss_val/{:.3f}--lr/{:.8f}".format(
                            step, loss.item(), loss_val, self.optimizer.param_groups[0]['lr']))
                        if self.loss_best > loss_val:
                            self.loss_best = loss_val
                            print("The current loss is {:.3f}, and store the weights of the model.\n".format(self.loss_best))
                            if warmup_state:
                                self.state_dict = copy(self.model.state_dict())
                            else:
                                checkpoint_file = osp.join(self.checkpoint, 'model.best.pth')
                                torch.save({'model_state_dict': self.model.state_dict()}, checkpoint_file)
                            index = step
                    if step > train_step:
                        print("Training stop. The time point for saving the best model is at {}.\n".format(index))
                        return self.loss_best, self.state_dict
                    step += 1
        return self.loss_best, self.state_dict

    def validation(self):
        """validate the performance of the trained model."""
        self.model.eval()
        if torch.cuda.device_count() > 1:
            self.model.module.set_eval()
        else:
            self.model.set_eval()
        loss_all = []
        for count in range(self.va_count):
            with h5py.File(self.data_dir + '/va_{}.hdf5'.format(count + 1), 'r') as f:
                sequence = np.array(f['seq'])
                sequence_rc = np.array(f['seq_rc'])
                dnase = np.array(f['DNase'])
                signal = np.array(f['signal'])
                position = np.array(f['position'])
                expression = np.array(f['expr'])
                rna_feat = np.array(f['rna'])
                if self.loop:
                    loop_input = np.array(f['loop1'])
                    loop_weight = np.array(f['loop2'])
                else:
                    loop_input = None
                    loop_weight = None
            va_loader = DataLoader(SourceDataSet(sequence, sequence_rc, dnase, signal, position, expression, rna_feat, loop_input, loop_weight), 
                                       batch_size=self.batch_size, shuffle=False)
            for _, sample_batch in enumerate(va_loader):
                seq = sample_batch["seq"].long().to(self.device)
                seq_rc = sample_batch["seq_rc"].long().to(self.device)
                dnase = sample_batch["dnase"].float().to(self.device)
                multi_signal = sample_batch["multi-signal"].float().to(self.device)
                position = sample_batch["position"].float().to(self.device)
                expr = sample_batch["expr"].float().to(self.device)
                rna_feat = sample_batch["rna"].float().to(self.device)
                if self.loop:
                    loop_in = sample_batch["loop1"].float().to(self.device)
                    loop_w = sample_batch["loop2"].float().to(self.device)
                else:
                    loop_in = None
                    loop_w = None
                seqs = {'seq': seq, 'seq_rc': seq_rc}
                signals = {'DNase': dnase, 'multi-signal': multi_signal, 'rna_feat': rna_feat}
                with torch.no_grad():
                    preds = self.model(seqs, signals, position, loop_in, loop_w)
                loss = self.criterion.fusion(preds[0], expr, preds[1])
                # loss = self.criterion.l1_loss(preds[0], expr)
                # loss = self.criterion.l1_loss_kl(preds[0], expr, preds[-1], position)
                loss_all.append(loss.item())
        # set the train state after validation
        self.model.train()
        if torch.cuda.device_count() > 1:
            self.model.module.set_train()
        else:
            self.model.set_train()
        return np.mean(loss_all)

    def test(self):
        """test the performance of the trained model."""
        self.model.eval()
        if torch.cuda.device_count() > 1:
            self.model.module.set_eval()
        else:
            self.model.set_eval()
        p_all = []
        t_all = []
        for count in range(self.te_count):
            with h5py.File(self.data_dir + '/te_{}.hdf5'.format(count + 1), 'r') as f:
                sequence = np.array(f['seq'])
                sequence_rc = np.array(f['seq_rc'])
                dnase = np.array(f['DNase'])
                signal = np.array(f['signal'])
                position = np.array(f['position'])
                expression = np.array(f['expr'])
                rna_feat = np.array(f['rna'])
                if self.loop:
                    loop_input = np.array(f['loop1'])
                    loop_weight = np.array(f['loop2'])
                else:
                    loop_input = None
                    loop_weight = None
            te_loader = DataLoader(SourceDataSet(sequence, sequence_rc, dnase, signal, position, expression, rna_feat, loop_input, loop_weight), 
                                       batch_size=self.batch_size, shuffle=False)
            for step, sample_batch in enumerate(te_loader):
                seq = sample_batch["seq"].long().to(self.device)
                seq_rc = sample_batch["seq_rc"].long().to(self.device)
                dnase = sample_batch["dnase"].float().to(self.device)
                multi_signal = sample_batch["multi-signal"].float().to(self.device)
                position = sample_batch["position"].float().to(self.device)
                expr = sample_batch["expr"].float().to(self.device)
                rna_feat = sample_batch["rna"].float().to(self.device)
                if self.loop:
                    loop_in = sample_batch["loop1"].float().to(self.device)
                    loop_w = sample_batch["loop2"].float().to(self.device)
                else:
                    loop_in = None
                    loop_w = None
                seqs = {'seq': seq, 'seq_rc': seq_rc}
                signals = {'DNase': dnase, 'multi-signal': multi_signal, 'rna_feat': rna_feat}
                with torch.no_grad():
                    preds = self.model(seqs, signals, position, loop_in, loop_w)
                pred = preds[0].view(-1).data.cpu().numpy()
                expr = expr.view(-1).data.cpu().numpy()
                if count == 0 and step == 0:
                    p_all = pred
                    t_all = expr
                else:
                    p_all = np.concatenate((p_all, pred))
                    t_all = np.concatenate((t_all, expr))
        pr = pearsonr(t_all, p_all)[0]
        mae = mean_absolute_error(t_all, p_all)
        print("pearson: {:.3f}\tmae: {:.3f}\n".format(pr, mae))
        # plot
        out_f = osp.join(self.checkpoint, 'pr.png')
        scatterplot(p_all, t_all, out_f)

        return pr, mae
