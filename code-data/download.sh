#!/usr/bin/bash

ROOT=/your/path

threadnum=2
tmp="/tmp/$$.fifo"
mkfifo ${tmp}
exec 6<> ${tmp}
rm ${tmp}
for((i=0; i<${threadnum}; i++))
do
    echo ""
done >&6

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
for CELL in "${CELLs[@]}"
do
    read -u6
   {
    cell=$(echo "$CELL" | tr ' ' '.')
    echo "Working on ${cell}."
    if [ ! -d ${ROOT}/Human-RawData/${cell} ]; then
            mkdir -p ${ROOT}/Human-RawData/${cell}/download
    fi
    python ${ROOT}/code-data/download.py -i ${ROOT}/code-data/Human_data_list/"${cell}.txt" -o ${ROOT}/Human-RawData/${cell}/download

    echo "" >&6
   }&
done
wait
exec 6>&-

