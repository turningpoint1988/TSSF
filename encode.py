# coding:utf-8
import os.path as osp
import re
import sys
import numpy as np
from Bio import SeqIO
import pyBigWig
import h5py
import argparse
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


def readloop(loopfile, bin=5000):
    loop_dict = {}
    with open(loopfile) as f:
        for line in f:
            line_split = line.strip().split('\t')
            chrom = line_split[0]
            if chrom not in loop_dict.keys():
                loop_dict[chrom] = {}
            fragment1_s = int(line_split[1]) // bin
            fragment2 = re.split(':|-|,', line_split[-1])
            fragment2_s = int(fragment2[1]) // bin
            value = float(fragment2[-1])
            if np.isinf(value):
                continue
            if fragment1_s <= fragment2_s:
                key = '{}-{}'.format(fragment1_s, fragment2_s)
            else:
                key = '{}-{}'.format(fragment2_s, fragment1_s)
            if key not in loop_dict[chrom]:
                loop_dict[chrom][key] = value
    return loop_dict


def retrieve_loop(distance, loop_dict, bin=5000):
    distance = np.array(distance, dtype=np.int64)
    distance = np.around(distance / bin)
    length = len(distance)
    interaction = np.zeros((length, length))
    # calculate interactions of all elements
    for i in range(length):
        fragment1_s = int(distance[i])
        for j in range(length):
            fragment2_s = int(distance[j])
            if i == j: continue
            if fragment1_s <= fragment2_s:
                key = '{}-{}'.format(fragment1_s, fragment2_s)
            else:
                key = '{}-{}'.format(fragment2_s, fragment1_s)
            if key in loop_dict.keys():
                value = loop_dict[key]
                value = np.log10(1+value)
            else:
                value = 0
            interaction[i, j] = round(value, 3)
    
    return interaction


def upload_hic(hic_dir, bin=1000):
    hic_dict = {}
    for chrom in INDEX:
        hic_file = hic_dir + '/{}.bedpe'.format(chrom)
        with open(hic_file) as f:
            lines = f.readlines()
        hic_dict[chrom] = {}
        for line in lines:
            line_split = line.strip().split('\t')
            fragment1_s = int(line_split[0]) // bin
            fragment2_s = int(line_split[1]) // bin
            contact = float(line_split[2])
            if contact < 30:
                continue
            if fragment1_s < fragment2_s:
                key = "{}-{}".format(fragment1_s, fragment2_s)
            elif fragment1_s > fragment2_s:
                key = "{}-{}".format(fragment2_s, fragment1_s)
            else:
                continue
            if key not in hic_dict[chrom].keys():
                hic_dict[chrom][key] = contact
        print("chromosome {} finished, total find {} valid contacts.".format(chrom, len(hic_dict[chrom])))
    return hic_dict


def retrieve_hic(distance, hic_dict, bin=1000):
    distance = np.array(distance, dtype=np.int64)
    distance = np.around(distance / bin)
    length = len(distance)
    interaction = np.zeros((length, length))
    for i in range(length):
        fragment1_s = int(distance[i])
        for j in range(length):
            fragment2_s = int(distance[j])
            if i == j: continue
            if fragment1_s < fragment2_s:
                key = '{}-{}'.format(fragment1_s, fragment2_s)
            else:
                key = '{}-{}'.format(fragment2_s, fragment1_s)
            if key in hic_dict.keys():
                contact = hic_dict[key]
                contact = np.log10(1+contact)
            else:
                contact = 0
            interaction[i, j] = round(contact, 3)
    return interaction


def datasplitLoop(gene_set, sequence_dict, tracks, split, splice, data_dir, out_dir, status, loop_dict1, loop_dict2):
    gene_name_all, seq_all, seq_rc_all, dnase_all, signal_all, exp_all, dis_all, rnaFeat_all, inter_all_1, inter_all_2 = [], [], [], [], [], [], [], [], [], []
    gene_index = 0
    for key, value in gene_set.items():
        chrom, TSS, _, expression, _ = value['info']
        if chrom not in split:
            continue
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
        # retrival loops
        interaction_1 = retrieve_hic(distances, loop_dict1[chrom], bin=1000)
        inter_all_1.append(interaction_1)
        interaction_2 = retrieve_loop(distances, loop_dict2[chrom])
        inter_all_2.append(interaction_2)
        # 
        gene_name_all.append(key.encode())
        seq_all.append(seqs)
        seq_rc_all.append(seqs_rc)
        dnase_all.append(dnases)
        signal_all.append(signals)
        exp_all.append([expression])
        dis_all.append(distances)
        rnaFeat_all.append(rnaFeat)
        gene_index += 1
        if gene_index % splice == 0:
            # save data
            gene_name_all = np.array(gene_name_all)
            seq_all = np.array(seq_all)
            seq_rc_all = np.array(seq_rc_all)
            dnase_all = np.array(dnase_all) 
            signal_all = np.array(signal_all)
            exp_all = np.array(exp_all)
            dis_all = np.array(dis_all)
            rnaFeat_all = np.array(rnaFeat_all)
            inter_all_1 = np.array(inter_all_1)
            inter_all_1[np.isinf(inter_all_1)] = 0
            inter_all_1[np.isnan(inter_all_1)] = 0
            inter_all_2 = np.array(inter_all_2)
            inter_all_2[np.isinf(inter_all_2)] = 0
            inter_all_2[np.isnan(inter_all_2)] = 0
            out_f = out_dir + '/{}_{}.hdf5'.format(status, int(np.ceil(gene_index / splice)))
            outputHDF5(gene_name_all, seq_all, seq_rc_all, dnase_all, signal_all, exp_all, dis_all, rnaFeat_all, inter_all_1, inter_all_2, out_f)
            # set default values
            gene_name_all, seq_all, seq_rc_all, dnase_all, signal_all, exp_all, dis_all, rnaFeat_all, inter_all_1, inter_all_2 = [], [], [], [], [], [], [], [], [], []
    if len(gene_name_all) > 0:
        # save data
        gene_name_all = np.array(gene_name_all)
        seq_all = np.array(seq_all)
        seq_rc_all = np.array(seq_rc_all)
        dnase_all = np.array(dnase_all) 
        signal_all = np.array(signal_all)
        exp_all = np.array(exp_all)
        dis_all = np.array(dis_all)
        rnaFeat_all = np.array(rnaFeat_all)
        inter_all_1 = np.array(inter_all_1)
        inter_all_1[np.isinf(inter_all_1)] = 0
        inter_all_1[np.isnan(inter_all_1)] = 0
        inter_all_2 = np.array(inter_all_2)
        inter_all_2[np.isinf(inter_all_2)] = 0
        inter_all_2[np.isnan(inter_all_2)] = 0
        out_f = out_dir + '/{}_{}.hdf5'.format(status, int(np.ceil(gene_index / splice)))
        outputHDF5(gene_name_all, seq_all, seq_rc_all, dnase_all, signal_all, exp_all, dis_all, rnaFeat_all, inter_all_1, inter_all_2, out_f)


def outputHDF5(gene_names, seqs, seqs_rc, dnase, signals, exp, dis, rnaFeat, inter_1, inter_2, out_f):
    print('sequence shape: {}\tdnase shape: {}\tsignals shape: {}\n'.format(seqs.shape, dnase.shape, signals.shape))
    print('rnaFeat shape: {}\tinter_1 shape: {}\tinter_2 shape: {}\n'.format(rnaFeat.shape, inter_1.shape, inter_2.shape))
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
        if inter_1 is not None:
            f.create_dataset('loop1', data=inter_1, **comp_kwargs)
        if inter_2 is not None:
            f.create_dataset('loop2', data=inter_2, **comp_kwargs)


def datasplit(gene_set, sequence_dict, tracks, split, splice, data_dir, out_dir, status):
    gene_name_all, seq_all, seq_rc_all, dnase_all, signal_all, exp_all, dis_all, rnaFeat_all = [], [], [], [], [], [], [], []
    gene_index = 0
    for key, value in gene_set.items():
        chrom, TSS, strand, expression, gene_id = value['info']
        if chrom not in split:
            continue
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
        gene_index += 1
        if gene_index % splice == 0:
            # save data
            gene_name_all = np.array(gene_name_all)
            seq_all = np.array(seq_all)
            seq_rc_all = np.array(seq_rc_all)
            dnase_all = np.array(dnase_all) 
            signal_all = np.array(signal_all)
            exp_all = np.array(exp_all)
            dis_all = np.array(dis_all)
            rnaFeat_all = np.array(rnaFeat_all)
            out_f = out_dir + '/{}_{}.hdf5'.format(status, int(np.ceil(gene_index / splice)))
            outputHDF5(gene_name_all, seq_all, seq_rc_all, dnase_all, signal_all, exp_all, dis_all, rnaFeat_all, None, None, out_f)
            # set default values
            gene_name_all, seq_all, seq_rc_all, dnase_all, signal_all, exp_all, dis_all, rnaFeat_all = [], [], [], [], [], [], [], []
    if len(gene_name_all) > 0:
        # save data
        gene_name_all = np.array(gene_name_all)
        seq_all = np.array(seq_all)
        seq_rc_all = np.array(seq_rc_all)
        dnase_all = np.array(dnase_all) 
        signal_all = np.array(signal_all)
        exp_all = np.array(exp_all)
        dis_all = np.array(dis_all)
        rnaFeat_all = np.array(rnaFeat_all)
        out_f = out_dir + '/{}_{}.hdf5'.format(status, int(np.ceil(gene_index / splice)))
        outputHDF5(gene_name_all, seq_all, seq_rc_all, dnase_all, signal_all, exp_all, dis_all, rnaFeat_all, None, None, out_f)


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Encoding data.")

    parser.add_argument("-r", dest="root", type=str, default="/data/zhangqinhu/gene-expression-level")
    parser.add_argument("-t", dest="target", type=str, default="K562")
    parser.add_argument("-o", dest="out_dir", type=str, default=None)
    parser.add_argument("-n", dest="number", type=int, default=10)

    return parser.parse_args()


SEQ_LEN = 600
WINDOW = 600
INDEX = ['chr' + str(i + 1) for i in range(22)] + ['chrX']
args = get_args()
ROOT = args.root
NUMBER = args.number
mRNA_df = pd.read_csv('/ROOT/mRNA_halflife_features.csv', index_col='Gene name')


def main():
    target = args.target
    genome = ROOT + '/Genome'
    data_dir = ROOT + '/Human-RawData/{}'.format(target)
    cCRE_dir = ROOT + '/cCREs'
    out_dir = args.out_dir
    #
    gene_file = data_dir + '/gene_expression.tsv'
    cCRE_file = cCRE_dir + '/hg38-cCREs.bed'
    gene_set = gene_selection(gene_file, cCRE_file)
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genome + '/hg38.fa'), 'fasta'))
    tracks = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']
    split_te = ['chr8', 'chr9']
    split_va = ['chr16']
    split_tr = list(set(INDEX)-set(split_te+split_va))
    #
    splice = 3000 # cCRE=10, splice=18000; cCRE=50, splice=3000
    loop_flag = False
    if target in ['K562', 'GM12878']:
        Loop_dir = ROOT + '/HiC/{}'.format(target)
        loop_dict1 = upload_hic(Loop_dir)
        Loop_dir = ROOT + '/Loops/{}'.format(target)
        loop_dict2 = readloop(Loop_dir + '/H3K27ac.5kb.longrange.bed')
        loop_flag = True
    if loop_flag:
        datasplitLoop(gene_set, sequence_dict, tracks, split_va, splice, data_dir, out_dir, 'va', loop_dict1, loop_dict2)
        datasplitLoop(gene_set, sequence_dict, tracks, split_te, splice, data_dir, out_dir, 'te', loop_dict1, loop_dict2)
        datasplitLoop(gene_set, sequence_dict, tracks, split_tr, splice, data_dir, out_dir, 'tr', loop_dict1, loop_dict2)
    else:
        datasplit(gene_set, sequence_dict, tracks, split_va, splice, data_dir, out_dir, 'va')
        datasplit(gene_set, sequence_dict, tracks, split_te, splice, data_dir, out_dir, 'te')
        datasplit(gene_set, sequence_dict, tracks, split_tr, splice, data_dir, out_dir, 'tr')
    

if __name__ == '__main__':  main()
