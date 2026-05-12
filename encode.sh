#!/bin/bash

ROOT=/your/path
NUM=10

threadnum=2
tmp="/tmp/$$.fifo"
mkfifo ${tmp}
exec 6<> ${tmp}
rm ${tmp}
for((i=0; i<${threadnum}; i++))
do
    echo ""
done >&6

# CELLs=('transverse colon' 'sigmoid colon' 'CD14-positive monocyte' 'dorsolateral prefrontal cortex' \
#       'HepG2' 'H9' 'colonic mucosa' 'kidney' 'stomach' 'testis' 'heart right ventricle' \
#       'keratinocyte' 'lower lobe of left lung' 'lower lobe of right lung' 'Peyer patch' 'fibroblast of lung' \
#       'psoas muscle' 'thoracic aorta' 'IMR-90' 'esophagus squamous epithelium' 'uterus' 'CD8-positive alpha-beta T cell' \
#       'adrenal gland' 'vagina' 'upper lobe of right lung' 'OCI-LY7' 'osteoblast' 'HeLa-S3' 'CD4-positive alpha-beta T cell' 'left lung' \
#       'liver' 'endothelial cell of umbilical vein' 'HCT116' 'A673' 'PC-3' 'right lobe of liver' 'astrocyte' 'SK-N-SH' 'natural killer cell' \
#       'prostate gland' 'mammary epithelial cell' 'ovary' 'left ventricle myocardium inferior' 'spleen' 'heart left ventricle' 'AG04450' 'foreskin fibroblast' \
#       'B cell' 'gastrocnemius medialis' 'GM23338' 'upper lobe of left lung' 'pancreas' 'skeletal muscle myoblast' 'right atrium auricular region' \
#       'body of pancreas' 'suprapubic skin' 'esophagus muscularis mucosa' 'ascending aorta' 'lung' 'tibial nerve' 'PC-9' 'thyroid gland' 'placenta' 'Panc1' 'MCF-7' 'fibroblast of dermis' 'breast epithelium' 'H1')
# CELLs=('CH12.LX' 'MEL')
CELLs=('GM12878' 'K562')

for CELL in "${CELLs[@]}"
do
  read -u6
  {  
    target=$(echo "$CELL" | tr ' ' '.')
    echo "Working on ${target}." 
    if [ ! -d ${ROOT}/Human-Data${NUM}/${target} ]; then
        mkdir -p ${ROOT}/Human-Data${NUM}/${target}
    fi
    python ${ROOT}/encode.py -r ${ROOT} -t ${target} \
                           -o ${ROOT}/Human-Data${NUM}/${target} \
                           -n ${NUM}

    echo "" >&6
   }&
done
wait
exec 6>&-



