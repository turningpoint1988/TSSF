#!/usr/bin/bash


ROOT=/your/path/TSSF
NUM=10

for target in $(ls ${ROOT}/Human-Data${NUM}/)
do
    echo "Working on ${target}."
    if [ ! -d ${ROOT}/models_human_${NUM}/${target} ]; then
        mkdir -p ${ROOT}/models_human_${NUM}/${target}
    fi
    
    echo ">> Starting to running the model. <<"
    python ${ROOT}/run.py -d ${ROOT}/Human-Data${NUM}/${target} \
                        -n ${target} \
                        -c ${ROOT}/models_human_${NUM}/${target} \
                        -g "0,1" \
                        -f 4

    echo ">> Running finished.<<"

done
