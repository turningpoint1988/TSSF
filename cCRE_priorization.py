#!/usr/bin/python
import re
import sys
import argparse
import numpy as np
import os.path as osp
from Bio import SeqIO
import pyBigWig
import json

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# custom functions defined by user
from model import TransformerT5, RefineT5
from utils import SourceDataSet
import h5py
import pandas as pd


def embed(seq):
    nucleotides = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    rc_dict = {'A': 'T', 'G': 'C',
               'C': 'G', 'T': 'A'}
    temp = []
    for c in seq:
        temp.append(nucleotides.get(c, 4))
    temp = np.array(temp)
    temp_rc = []
    for c in seq[::-1]:
        c_rc = rc_dict.get(c, 'N')
        temp_rc.append(nucleotides.get(c_rc, 4)) 
    temp_rc = np.array(temp_rc)

    return temp, temp_rc


def getbigwig(file, chrom, start, end):
    bw = pyBigWig.open(file)
    sample = np.array(bw.values(chrom, start, end))
    bw.close()
    return sample


def extract_track(data_dir, tracks, chrom, start, end):
    signals = []
    dnase = 0
    for track in tracks:
        signal = getbigwig(data_dir + '/{}_merged.bigWig'.format(track), chrom, start, end)
        signal[np.isnan(signal)] = 0.
        signal = np.log10(1+signal)
        if track == 'DNase':
            dnase = signal
        signal_mean = np.mean(signal)
        signals.append(round(signal_mean, 3))

    return dnase, signals


def gene_selection(gene_file, cCRE_file, merge_flag=True):
    cCRE_set = {}
    with open(cCRE_file) as f:
        for line in f:
            line_split = line.strip().split('\t')
            chrom = line_split[0]
            start = int(line_split[1])
            end = int(line_split[2])
            mid = (start + end) // 2
            if chrom not in cCRE_set.keys():
                cCRE_set[chrom] = [mid]
            else:
                cCRE_set[chrom] += [mid]
    # merge adjacent regions
    if merge_flag:
        for chrom in cCRE_set.keys():
            cCRE_pos = cCRE_set[chrom]
            cCRE_pos.sort()
            print("The number of cCREs on {} is {}".format(chrom, len(cCRE_set[chrom])))
            cCRE_pos_new = []
            start = 0
            end = start + 1
            while start < len(cCRE_pos):
                pos_start = cCRE_pos[start]
                while end < len(cCRE_pos):
                    pos_end = cCRE_pos[end]
                    if pos_end - pos_start > WINDOW:
                        tmp = cCRE_pos[start:end]
                        tmp = tmp[len(tmp)//2]
                        cCRE_pos_new.append(tmp)
                        start = end
                        end = start + 1
                        break
                    else:
                        end += 1
                if end >= len(cCRE_pos):
                    tmp = cCRE_pos[start:end]
                    tmp = tmp[len(tmp)//2]
                    cCRE_pos_new.append(tmp)
                    start = end
            cCRE_set[chrom] = cCRE_pos_new
            print("The number of merged cCREs on {} is {}".format(chrom, len(cCRE_set[chrom])))
    # selecting cCREs around genes (TSS) according to their distances
    genes = {}
    with open(gene_file) as f:
        for line in f:
            line_split = line.strip().split('\t')
            chrom = line_split[0]
            start = int(line_split[1])
            end = int(line_split[2])
            strand = line_split[3]
            gene_type = line_split[4]
            gene_name = line_split[5]
            gene_id = line_split[6]
            expression = float(line_split[-1])
            if chrom not in INDEX:
                continue
            # appoint a gene type
            if gene_type != 'protein_coding':
                continue
            if strand == '+':
                TSS = start
            elif strand == '-':
                TSS = end
            else:
                print("no exact direction.")
                sys.exit(0)
            # compute the distances between genes' TSS and cCREs
            cCRE_pos = np.array(cCRE_set[chrom], dtype=np.int64)
            distance = cCRE_pos - TSS
            left_shift = right_shift = 0
            # the left direction
            left_index = (distance < 0)
            left_dis = distance[left_index]
            left_pos = cCRE_pos[left_index]
            index = np.argsort(left_dis)
            index = index[::-1]
            left_pos_retained = left_pos[index]
            if len(index) < NUMBER:
                right_shift = NUMBER - len(index)
            # the right direction
            right_index = (distance > 0)
            right_dis = distance[right_index]
            right_pos = cCRE_pos[right_index]
            index = np.argsort(right_dis)
            right_pos_retained = right_pos[index]
            if len(index) < NUMBER:
                left_shift = NUMBER - len(index)
            # trim
            left_pos_retained = left_pos_retained[:(NUMBER+left_shift)]
            left_pos_retained = left_pos_retained[::-1]
            right_pos_retained = right_pos_retained[:(NUMBER+right_shift)]
            genes[gene_name] = {'info': [chrom, TSS, strand, expression, gene_id],
                                'cCRE_left_pos': left_pos_retained,
                                'cCRE_right_pos': right_pos_retained}
    return genes           


def MGPUtoSingle(state_dict):
    from collections import OrderedDict
    
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:]  # delete `module.`
        name = k.replace("module.", "")
        state_dict_new[name] = v
        
    return state_dict_new


def determine_index(gene_dict, target_gene_dict):
    gene_ccre = {}
    index_label_dict = {}
    for gene, loc_set in target_gene_dict.items():
        gene_name = gene
        if gene_name not in gene_dict.keys():
            print("the gene {} is not existed.".format(gene_name))
            continue
        if len(loc_set) == 0:
            continue
        chrom, TSS, _, expression, _ = gene_dict[gene_name]['info']
        left_pos_retained = gene_dict[gene_name]['cCRE_left_pos']
        right_pos_retained = gene_dict[gene_name]['cCRE_right_pos']
        pos_retained = np.concatenate((np.array([TSS]), left_pos_retained, right_pos_retained))
        #
        num_left = len(left_pos_retained)
        num_right = len(right_pos_retained)
        index_label_b = []
        index_label_m = []
        index_label_h = []
        index_label_s = []
        for loc in loc_set:
            mid = loc[1]
            label = loc[2]
            dis = np.abs(pos_retained-mid)
            index = np.argsort(dis)
            if index[0] == 0:
                index_i = index[1]
            else:
                index_i = index[0]
            distance = dis[index_i] // 1000
            if distance > 1:
                continue
            pos_retained[index_i] = mid
            distance = np.abs(mid-TSS)//1000
            if distance <= THRESHOLDS[0]:
                index_label_b.append((int(index_i), int(label)))
            elif THRESHOLDS[0] < distance <= THRESHOLDS[1]:
                index_label_m.append((int(index_i), int(label)))
            elif THRESHOLDS[1] < distance <= THRESHOLDS[2]:
                index_label_h.append((int(index_i), int(label)))
            else:
                index_label_s.append((int(index_i), int(label)))
        gene_ccre[gene_name] = {'info': [chrom, TSS, expression],
                                'cCRE_left_pos': pos_retained[1:(num_left+1)],
                                'cCRE_right_pos': pos_retained[(num_left+1):]}
        index_label_dict[gene_name] = {'index_label_b': index_label_b,
                                       'index_label_m': index_label_m,
                                       'index_label_h': index_label_h,
                                       'index_label_s': index_label_s}
    return gene_ccre, index_label_dict


def encode_data(gene_set, sequence_dict, tracks, data_dir, out_f):
    gene_name_all, seq_all, seq_rc_all, dnase_all, signal_all, exp_all, dis_all, rnaFeat_all = [], [], [], [], [], [], [], []
    for key, value in gene_set.items():
        chrom, TSS, expression = value['info']
        cCRE_left_pos = value['cCRE_left_pos']
        cCRE_right_pos = value['cCRE_right_pos']
        # extract the gene and corresponding cCREs, and signals from tracks
        seqs = []
        seqs_rc = []
        dnases = []
        signals = []
        distances = []
        # the gene interval
        start = TSS - SEQ_LEN // 2
        end = TSS + int(np.ceil(SEQ_LEN/2))
        gene = str(sequence_dict[chrom].seq[start:end]).upper()
        seq, seq_rc = embed(gene)
        seqs.append(seq)
        seqs_rc.append(seq_rc)
        dnase, signal = extract_track(data_dir, tracks, chrom, start, end)  # 4xL
        dnases.append(dnase)
        signals.append(signal)
        distances.append(TSS)
        # the cCREs on the left
        for mid in cCRE_left_pos:
            start = mid - SEQ_LEN // 2
            end = mid + int(np.ceil(SEQ_LEN/2))
            cCRE = str(sequence_dict[chrom].seq[start:end]).upper()
            seq, seq_rc = embed(cCRE)
            seqs.append(seq)
            seqs_rc.append(seq_rc)
            dnase, signal = extract_track(data_dir, tracks, chrom, start, end)  # 4xL
            dnases.append(dnase)
            signals.append(signal)
            distances.append(mid)
        # the cCREs on the right
        for mid in cCRE_right_pos:
            start = mid - SEQ_LEN // 2
            end = mid + int(np.ceil(SEQ_LEN/2))
            cCRE = str(sequence_dict[chrom].seq[start:end]).upper()
            seq, seq_rc = embed(cCRE)
            seqs.append(seq)
            seqs_rc.append(seq_rc)
            dnase, signal = extract_track(data_dir, tracks, chrom, start, end)  # 4xL
            dnases.append(dnase)
            signals.append(signal)
            distances.append(mid)
        # retrieve mRNA features
        try:
            rnaFeat = list(mRNA_df.loc[key][['UTR5LEN_log10zscore','CDSLEN_log10zscore','INTRONLEN_log10zscore',
                                                'UTR3LEN_log10zscore','UTR5GC','CDSGC','UTR3GC', 'ORFEXONDENSITY']].values.astype(float))
        except:
            rnaFeat = [0] * 8
        # split data into tr, va, te
        gene_name_all.append(key.encode())
        seq_all.append(seqs)
        seq_rc_all.append(seqs_rc)
        dnase_all.append(dnases)
        signal_all.append(signals)
        exp_all.append([expression])
        dis_all.append(distances)
        rnaFeat_all.append(rnaFeat)
    # save data
    gene_name_all = np.array(gene_name_all)
    seq_all = np.array(seq_all)
    seq_rc_all = np.array(seq_rc_all)
    dnase_all = np.array(dnase_all) 
    signal_all = np.array(signal_all)
    exp_all = np.array(exp_all)
    dis_all = np.array(dis_all)
    rnaFeat_all = np.array(rnaFeat_all)
    outputHDF5(gene_name_all, seq_all, seq_rc_all, dnase_all, signal_all, exp_all, dis_all, rnaFeat_all, out_f)


def outputHDF5(gene_names, seqs, seqs_rc, dnase, signals, exp, dis, rnaFeat, out_f):
    print('sequence shape: {}\tdnase shape: {}\tsignals shape: {}\n'.format(seqs.shape, dnase.shape, signals.shape))
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    dt = h5py.special_dtype(vlen=str)
    with h5py.File(out_f, 'w') as f:
        f.create_dataset('gene', data=gene_names,  dtype=dt, **comp_kwargs)
        f.create_dataset('seq', data=seqs, **comp_kwargs)
        f.create_dataset('seq_rc', data=seqs_rc, **comp_kwargs)
        f.create_dataset('DNase', data=dnase, **comp_kwargs)
        f.create_dataset('signal', data=signals, **comp_kwargs)
        f.create_dataset('expr', data=exp, **comp_kwargs)
        f.create_dataset('position', data=dis, **comp_kwargs)
        f.create_dataset('rna', data=rnaFeat, **comp_kwargs)


def normalization(x, flag=True):
    x_part = x[1:]
    x_max = np.max(x_part)
    x_min = np.min(x_part)
    if flag:
        x_norm = (x - x_min) / (x_max - x_min)
    else:
        x_norm = (x - x_min) / (x_max - x_min) * 2 - 1
    x_norm[0] = 0
    return x_norm


def activity(chrom, loc, target):
    activity = 1
    data_dir = '/ROOT/Human-RawData/{}'.format(target)
    start = loc - 300
    end = loc + 300
    for track in ['DNase', 'H3K27ac']:
        feature = getbigwig(data_dir + '/{}_merged.bigWig'.format(track), chrom, int(start.data.numpy()), int(end.data.numpy()))
        feature[np.isnan(feature)] = 0.
        activity *= np.sum(feature)
    
    return np.sqrt(activity)


def compute_attention(seqs, signals, position, index_label, chrom, target, flag='all'):
    # Load weights
    device = torch.device("cpu") 
    checkpoint_file = osp.join('/ROOT/models_human_50_sig/{}'.format(target), 'model.best.pth')
    chk = torch.load(checkpoint_file, map_location='cpu') # cuda:0
    state_dict = chk['model_state_dict']
    state_dict = MGPUtoSingle(state_dict)
    model = TransformerT5()
    model.load_state_dict(state_dict)
    model.to(device)
    # set eval state for Dropout and BN layers
    model.eval()
    with torch.no_grad():
        preds = model(seqs, signals, position, None, None)
    attention = [att.data.numpy() for att in preds[-1]]
    attention_all = []
    if flag == 'first':
        attention_all = attention[0][0]
    elif flag == 'last':
        attention_all = attention[-1][0]
    else:
        ## integrate all attentions across all layers
        for step, att in enumerate(attention):
            if step == 0:
                attention_all = att[0] 
            else:
                attention_all += att[0]
    _, m, _ = attention_all.shape
    att_a = np.mean(attention_all, axis=0) # m*m
    att_gene = att_a[0]
    att_gene = normalization(att_gene)
    score_all = []
    label_all = []
    for index, label in index_label:
        ## add ABC scores to attention weights
        # loc = position[0, index]
        # abc = activity(chrom, loc, target) / np.abs(loc-position[0, 0])
        # score = att_gene[index] * abc
        score = att_gene[index]
        score_all.append(score)
        label_all.append(label)
    
    return score_all, label_all


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="A two-stage framework for cCREs priorization.")

    parser.add_argument("-r", dest="root", type=str, default="/your/path/TSSF",
                        help="A directory containing the training data.")
    parser.add_argument("-t", dest="target", type=str, default="K562",
                        help="The name of a specified data.")

    return parser.parse_args()


THRESHOLDS = [10, 50, 100]
NUMBER = 50
SEQ_LEN = 600
WINDOW = 600
INDEX = ['chr' + str(i + 1) for i in range(22)] + ['chrX']
args = get_args()
ROOT = args.root
mRNA_df = pd.read_csv('/ROOT/mRNA_halflife/mRNA_halflife_features.csv', index_col='Gene name')


def main():
    target = args.target
    genome = ROOT + '/Genome'
    data_dir = ROOT + '/Human-RawData/{}'.format(target)
    cCRE_dir = ROOT + '/cCREs'
    gene_file = data_dir + '/gene_expression.tsv'
    cCRE_file = cCRE_dir + '/hg38-cCREs.bed'
    gene_set = gene_selection(gene_file, cCRE_file)
    # Fulco.hg38.bed; Integration.hg38.bed
    source = 'Integration'
    crispr_file = ROOT + '/CRISPR/{}.hg38.bed'.format(source)
    df = pd.read_csv(crispr_file, sep='\t')
    row, _ = df.shape
    target_gene = {}
    for i in range(row):
        chrom = df.loc[i]['chr']
        start = df.loc[i]['start']
        end = df.loc[i]['end']
        gene_name = df.loc[i]['Gene']
        label = df.loc[i]['Significant']
        mid = (start + end) // 2
        if gene_name not in target_gene.keys():
            target_gene[gene_name] = [(chrom, mid, label)]
        else:
            target_gene[gene_name] += [(chrom, mid, label)]
    
    target_gene_dict, index_label_dict = determine_index(gene_set, target_gene)
    out_f = ROOT + '/CRISPR/{}.json'.format(source)
    with open(out_f, 'w') as f:
        json.dump(index_label_dict, f)
    out_f = ROOT + '/CRISPR/{}.hdf5'.format(source)
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genome + '/hg38.fa'), 'fasta'))
    tracks = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']
    encode_data(target_gene_dict, sequence_dict, tracks, data_dir, out_f)
    #### load data
    with h5py.File(out_f, 'r') as f:
        gene_names = np.array(f['gene'])
        gene_names = [x.decode() for x in gene_names]
        sequence = np.array(f['seq'])
        sequence_rc = np.array(f['seq_rc'])
        dnase = np.array(f['DNase'])
        signal = np.array(f['signal'])
        position = np.array(f['position'])
        expression = np.array(f['expr'])
        rna_feat = np.array(f['rna'])
    device = torch.device("cpu") # cuda:0
    va_loader = DataLoader(SourceDataSet(sequence, sequence_rc, dnase, signal, position, expression, rna_feat, None, None), 
                                       batch_size=1, shuffle=False)
    ############ Classification of CRISPR-based data through attentions ##################
    bottom = [[],[]]
    mid = [[],[]]
    high = [[],[]]
    super = [[],[]]
    for i, sample_batch in enumerate(va_loader):
        seq = sample_batch["seq"].long().to(device)
        seq_rc = sample_batch["seq_rc"].long().to(device)
        dnase = sample_batch["dnase"].float().to(device)
        multi_signal = sample_batch["multi-signal"].float().to(device)
        position = sample_batch["position"].float().to(device)
        rna_feat = sample_batch["rna"].float().to(device)
        expr = sample_batch["expr"].float()
        gene_name = gene_names[i]
        chrom, _, _ = target_gene[gene_name][0]
        if expr < 0.1:
            print("Warning, the {}'s expression level {} is relatively low.".format(gene_name, expr))
            continue
        seqs = {'seq': seq, 'seq_rc': seq_rc}
        signals = {'DNase': dnase, 'multi-signal': multi_signal, 'rna_feat': rna_feat}
        flag = 'first'
        # bottom
        index_label_b = index_label_dict[gene_name]['index_label_b']
        if len(index_label_b) > 0:
            score, label = compute_attention(seqs, signals, position, index_label_b, chrom, target, flag)
            bottom[0] += score
            bottom[1] += label
        # middle
        index_label_m = index_label_dict[gene_name]['index_label_m']
        if len(index_label_m) > 0:
            score, label = compute_attention(seqs, signals, position, index_label_m, chrom, target, flag)
            mid[0] += score
            mid[1] += label
        # high
        index_label_h = index_label_dict[gene_name]['index_label_h']
        if len(index_label_h) > 0:
            score, label = compute_attention(seqs, signals, position, index_label_h, chrom, target, flag)
            high[0] += score
            high[1] += label
        # super
        index_label_s = index_label_dict[gene_name]['index_label_s']
        if len(index_label_s) > 0:
            score, label = compute_attention(seqs, signals, position, index_label_s, chrom, target, flag)
            super[0] += score
            super[1] += label
    #
    out_f = '/ROOT/results/{}_score.txt'.format(source)
    f = open(out_f, 'a')
    f.write("Running attention from RefineT5 ({}).\n".format(flag)) 
    #
    num_pos = np.sum(np.asarray(bottom[1]) == 1)
    num_neg = np.sum(np.asarray(bottom[1]) == 0)
    if num_pos > 0 and num_neg > 0:
        prauc = average_precision_score(bottom[1], bottom[0])
        rocauc = roc_auc_score(bottom[1], bottom[0])
        f.write("No. of pos and neg enhance-gene pairs in bottom is {} and {}\n".format(num_pos, num_neg))
        f.write("PRAUC is {} and ROCAUC is {}.\n".format(prauc, rocauc))
        print("No. of pos and neg enhance-gene pairs in bottom is {} and {}\n".format(num_pos, num_neg))
        print("PRAUC is {} and ROCAUC is {}.\n".format(prauc, rocauc))
    else:
        print("No. of pos and neg enhance-gene pairs in bottom is {} and {}\n".format(num_pos, num_neg))
    #
    num_pos = np.sum(np.asarray(mid[1]) == 1)
    num_neg = np.sum(np.asarray(mid[1]) == 0)
    prauc = average_precision_score(mid[1], mid[0])
    rocauc = roc_auc_score(mid[1], mid[0])
    if num_pos > 0 and num_neg > 0:
        f.write("No. of pos and neg enhance-gene pairs in mid is {} and {}\n".format(num_pos, num_neg))
        f.write("PRAUC is {} and ROCAUC is {}.\n".format(prauc, rocauc))
        print("No. of pos and neg enhance-gene pairs in mid is {} and {}\n".format(num_pos, num_neg))
        print("PRAUC is {} and ROCAUC is {}.\n".format(prauc, rocauc))
    else:
        print("No. of pos and neg enhance-gene pairs in mid is {} and {}\n".format(num_pos, num_neg))
    #
    num_pos = np.sum(np.asarray(high[1]) == 1)
    num_neg = np.sum(np.asarray(high[1]) == 0)
    prauc = average_precision_score(high[1], high[0])
    rocauc = roc_auc_score(high[1], high[0])
    if num_pos > 0 and num_neg > 0:
        f.write("No. of pos and neg enhance-gene pairs in high is {} and {}\n".format(num_pos, num_neg))
        f.write("PRAUC is {} and ROCAUC is {}.\n".format(prauc, rocauc))
        print("No. of pos and neg enhance-gene pairs in high is {} and {}\n".format(num_pos, num_neg))
        print("PRAUC is {} and ROCAUC is {}.\n".format(prauc, rocauc))
    else:
        print("No. of pos and neg enhance-gene pairs in high is {} and {}\n".format(num_pos, num_neg))
    #
    num_pos = np.sum(np.asarray(super[1]) == 1)
    num_neg = np.sum(np.asarray(super[1]) == 0)
    prauc = average_precision_score(super[1], super[0])
    rocauc = roc_auc_score(super[1], super[0])
    if num_pos > 0 and num_neg > 0:
        f.write("No. of pos and neg enhance-gene pairs in super is {} and {}\n".format(num_pos, num_neg))
        f.write("PRAUC is {} and ROCAUC is {}.\n".format(prauc, rocauc))
        print("No. of pos and neg enhance-gene pairs in super is {} and {}\n".format(num_pos, num_neg))
        print("PRAUC is {} and ROCAUC is {}.\n".format(prauc, rocauc))
    else:
        print("No. of pos and neg enhance-gene pairs in super is {} and {}\n".format(num_pos, num_neg))
    f.close()
    

if __name__ == "__main__":
    main()

