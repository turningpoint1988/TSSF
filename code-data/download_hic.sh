#!/usr/bin/bash

ROOT=/your/path

threadnum=1
tmp="/tmp/$$.fifo"
mkfifo ${tmp}
exec 6<> ${tmp}
rm ${tmp}
for((i=0; i<${threadnum}; i++))
do
    echo ""
done >&6

CELLs=('GM12878' 'K562')

for CELL in "${CELLs[@]}"
do
    read -u6
   {
      echo "Working on ${CELL}."
    
      python ${ROOT}/code-data/download_hic.py -r ${ROOT} -t ${CELL}

      echo "" >&6
   }&
done
wait
exec 6>&-
