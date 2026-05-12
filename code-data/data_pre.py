# coding:utf-8
import os.path as osp
import pandas as pd
import glob
import numpy as np
import argparse


def extract_gene(gencode_file):
    genes = {}
    with open(gencode_file) as f:
        lines = f.readlines()

    for line in lines[5:]:
        line_split = line.strip().split('\t')
        chrom = line_split[0]
        type = line_split[2]
        start = line_split[3]
        end = line_split[4]
        strand = line_split[6]
        info = line_split[8]
        info_split = info.split(';')
        gene_id = eval(info_split[0].strip().split()[-1])
        if type == 'gene' and gene_id not in genes.keys():
            gene_type = eval(info_split[1].strip().split()[-1])
            gene_name = eval(info_split[2].strip().split()[-1])
            head = '{}\t{}\t{}\t{}\t{}\t{}'.format(chrom, start, end, strand, gene_type, gene_name)
            genes[gene_id] = [head]
        if type == 'transcript':
            transcript_id = eval(info_split[1].strip().split()[-1])
            genes[gene_id] += [transcript_id]

    return genes


def extract_gene_expression(gene_expr_file):
    genes = {}
    dml_df = pd.read_csv(gene_expr_file, sep='\t')
    row, _ = dml_df.shape
    for i in range(row):
        gene_id = dml_df.loc[i, "gene_id"]
        tpm = dml_df.loc[i, "TPM"]
        if 'ENSM' not in gene_id:
            continue
        if gene_id not in genes.keys():
            genes[gene_id] = [tpm]
        else:
            genes[gene_id] += [tpm]
    
    return genes


def integrate_gene(in_files, gene_annotation, out_f):
    gene_set = []
    for each_file in in_files:
        print(each_file)
        gene_set.append(extract_gene_expression(each_file))

    f = open(out_f, 'w')
    for key, value in gene_annotation.items():
        gene_id = key
        info = value[0]
        expression = []
        for gene in gene_set:
            if gene_id not in gene.keys():
                continue
            expression += gene[gene_id]
        expression = np.log1p(np.mean(expression))
        f.write("{}\t{}\t{:.3f}\n".format(info, gene_id, expression))

    f.close()



def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Data preparation.")

    parser.add_argument("-r", dest="root", type=str, default="/your/path/",
                        help="A directory containing the training data.")
    parser.add_argument("-t", dest="target", type=str, default="K562",
                        help="A directory containing the training data.")

    return parser.parse_args()


def main():
    args = get_args()
    ROOT = args.root
    target = args.target
    # extract genes from gencode annotation
    # gencode.vM21.annotation.gtf; gencode.v29.annotation.gtf
    gencode = ROOT + '/GENCODE/gencode.v29.annotation.gtf'
    genes = extract_gene(gencode)
    # integrate multiple transcripts from the same cell line/type
    in_files = glob.glob(ROOT + '/Human-RawData/{}/download/*.gene.tsv'.format(target))
    out_f = ROOT + '/Human-RawData/{}/gene_expression.tsv'.format(target)
    integrate_gene(in_files, genes, out_f)


if __name__ == '__main__':  main()
