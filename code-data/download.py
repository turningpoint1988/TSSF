#!/usr/bin/env python

import os, argparse
import os.path as osp
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-i", dest="inputfile", type=str, default='')
    parser.add_argument("-o", dest="outdir", type=str, default='')

    return parser.parse_args()


def download(inputfile, outdir):
    dml_df = pd.read_csv(inputfile, sep='\t')
    row, _ = dml_df.shape
    for i in range(row):
        accession = dml_df.loc[i, "File accession"]
        assay = dml_df.loc[i, "Assay"]
        biosample_name = dml_df.loc[i, "Biosample term name"]
        target = dml_df.loc[i, "Experiment target"]
        url = dml_df.loc[i, "File download URL"]

        if 'RNA-seq' in assay:
            print("downloading gene quantification ({}) from {}...".format(accession, biosample_name))
            # url = 'https://www.encodeproject.org/files/{}/@@download/{}.{}'.format(accession, accession, format)
            download_url = url
            file_format = download_url.split('.')[-1]
            outfile = outdir + '/{}.gene.{}'.format(accession, file_format)
            if osp.exists(outfile):
                os.system('curl -s -S -L -C - -o {} {}'.format(outfile, download_url))
            else:
                os.system('curl -s -S -J -L -o {} {}'.format(outfile, download_url))
            
        if 'DNase' in assay:
            print("downloading DNase-seq track ({}) from {}...".format(accession, biosample_name))
            download_url = url
            file_format = download_url.split('.')[-1]
            outfile = outdir + '/{}.DNase.{}'.format(accession, file_format)
            if osp.exists(outfile):
                os.system('curl -s -S -L -C - -o {} {}'.format(outfile, download_url))
            else:
                os.system('curl -s -S -J -L -o {} {}'.format(outfile, download_url))
        
        if 'ChIP-seq' in assay:
            target_ = target.split('-')[0]
            print("downloading {} ({}) tracks from {}...".format(target_, accession, biosample_name))
            download_url = url
            file_format = download_url.split('.')[-1]
            outfile = outdir + '/{}.{}.{}'.format(accession, target_, file_format)
            if osp.exists(outfile):
                os.system('curl -s -S -L -C - -o {} {}'.format(outfile, download_url))
            else:
                os.system('curl -s -S -J -L -o {} {}'.format(outfile, download_url))


args = get_args()
download(args.inputfile, args.outdir)




