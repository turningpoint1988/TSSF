#!/bin/bash
# conda install bioconda::ucsc-bigwigmerge
# conda install bioconda::ucsc-bedgraphtobigwig
ROOT=/your/path/TSSF

# CELLs=('transverse colon' 'sigmoid colon' 'CD14-positive monocyte' \
#       'HepG2' 'H9' 'colonic mucosa' 'kidney' 'stomach' 'testis' 'heart right ventricle' \
#       'keratinocyte' 'lower lobe of left lung' 'lower lobe of right lung' 'Peyer patch' 'fibroblast of lung' \
#       'psoas muscle' 'thoracic aorta' 'IMR-90' 'esophagus squamous epithelium' 'uterus' 'CD8-positive alpha-beta T cell' \
#       'adrenal gland' 'vagina' 'upper lobe of right lung' 'OCI-LY7' 'osteoblast' 'HeLa-S3' 'CD4-positive alpha-beta T cell' 'left lung' \
#       'liver' 'endothelial cell of umbilical vein' 'HCT116' 'A673' 'PC-3' 'right lobe of liver' 'astrocyte' 'SK-N-SH' 'natural killer cell' \
#       'prostate gland' 'mammary epithelial cell' 'ovary' 'left ventricle myocardium inferior' 'spleen' 'heart left ventricle' 'AG04450' 'foreskin fibroblast' \
#       'B cell' 'gastrocnemius medialis' 'GM23338' 'upper lobe of left lung' 'pancreas' 'skeletal muscle myoblast' 'right atrium auricular region' \
#       'body of pancreas' 'suprapubic skin' 'esophagus muscularis mucosa' 'ascending aorta' 'lung' 'tibial nerve' 'PC-9' 'thyroid gland' 'placenta' 'Panc1' 'MCF-7' 'fibroblast of dermis' 'breast epithelium' 'H1')
CELLs=('GM12878' 'K562')
# CELLs=('CH12.LX' 'MEL')
tracks=('DNase' 'H3K4me3' 'H3K27ac' 'CTCF')

for CELL in "${CELLs[@]}"
do
    target=$(echo "$CELL" | tr ' ' '.')
    echo "Working on ${target}."    
    python /code/code-data/data_pre.py -r ${ROOT} -t ${target}
    for track in ${tracks[*]}
    do
        echo "merging ${track} tracks"
        num=$(ls ${ROOT}/Human-RawData/${target}/download | grep "${track}" | wc -l)
        if [ "${num}" -eq 1 ]; then
            file=$(ls ${ROOT}/Human-RawData/${target}/download | grep "${track}")
            echo "${file}"
            cp -f ${ROOT}/Human-RawData/${target}/download/${file} ${ROOT}/Human-RawData/${target}/${track}_merged.bigWig
        else
            bigWigMerge -max ${ROOT}/Human-RawData/${target}/download/*.${track}.bigWig ${ROOT}/Human-RawData/${target}/temp.bedGraph
            sort -k1,1 -k2,2n ${ROOT}/Human-RawData/${target}/temp.bedGraph > ${ROOT}/Human-RawData/${target}/temp_sorted.bedGraph
            # mm10.chrom.sizes; hg38.chrom.sizes
            bedGraphToBigWig ${ROOT}/Human-RawData/${target}/temp_sorted.bedGraph ${ROOT}/Genome/hg38.chrom.sizes ${ROOT}/Human-RawData/${target}/${track}_merged.bigWig
        
        fi
        rm -f ${ROOT}/Human-RawData/${target}/*.bedGraph
        
    done  
      
done



