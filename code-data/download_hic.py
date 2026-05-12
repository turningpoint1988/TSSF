#!/usr/bin/env python

import os, argparse
import os.path as osp
import hicstraw
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-r", dest="root", type=str, default='/your/path/TSSF')
    parser.add_argument("-t", dest="target", type=str, default='K562')

    return parser.parse_args()


def download(root, target):
    if target == 'K562':
        accession = "ENCFF080DPJ"
    elif target == 'GM12878':
        accession = "ENCFF053VBX"
    download_url = "https://www.encodeproject.org/files/{}/@@download/{}.hic".format(accession, accession)
    outfile = root + '/HiC/{}/{}.hic'.format(target, target)
    if osp.exists(outfile):
        os.system('curl -s -S -L -C - -o {} {}'.format(outfile, download_url))
    else:
        os.system('curl -s -S -J -L -o {} {}'.format(outfile, download_url))


def write_contact_from_hic(root, target, bin=1000):
    # apt update; apt install openjdk-8-jdk openjdk-8-jre
    # if not normalization, use the command: java -jar juicer_tools.jar addNorm ...
    # hg38; in situ Hi-C
    # K562: ENCFF080DPJ, ENCFF616PUW; GM12878: ENCFF053VBX, ENCFF555ISR
    if target == 'K562':
        accession = "ENCFF080DPJ"
    elif target == 'GM12878':
        accession = "ENCFF053VBX"
    # source_file = "https://www.encodeproject.org/files/{}/@@download/{}.hic".format(accession, accession)
    source_file = root + '/HiC/{}/{}.hic'.format(target, target)
    hic = hicstraw.HiCFile(source_file)
    print(hic.getChromosomes())
    print(hic.getGenomeID())
    print(hic.getResolutions())
    # hic.getMatrixZoomData(chrom1, chrom2, data_type, normalization, "BP", resolution)
    # mzd = hic.getMatrixZoomData('4', '4', "observed", "KR", "BP", bin)
    # numpy_matrix = mzd.getRecordsAsMatrix(10000000, 12000000, 10000000, 12000000)
    # normalization: NONE, VC, VC_SQRT, KR, SCALE
    chromosomes = ['chr' + str(i + 1) for i in range(22)] + ['chrX']
    for chrom in chromosomes:
        out_f = open(root+'/HiC/{}/{}.bedpe'.format(target, chrom), 'w')
        result = hicstraw.straw('observed', 'NONE', source_file, chrom, chrom, 'BP', bin)
        for i in range(len(result)):
            out_f.write("{}\t{}\t{}\n".format(result[i].binX, result[i].binY, result[i].counts))
        out_f.close()
    print("Writing finished.")


args = get_args()
download(args.root, args.target)
write_contact_from_hic(args.root, args.target)




