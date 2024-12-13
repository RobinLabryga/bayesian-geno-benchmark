#!/bin/bash

export ARCHDEFS="$PWD"/CUTEst/ARCHDefs/
export SIFDECODE="$PWD"/CUTEst/SIFDecode/
export MASTSIF="$PWD"/CUTEst/sif/
export CUTEST="$PWD"/CUTEst/CUTEst/
export MYARCH="pc64.lnx.gfo"

run_names=("1-1" "1-2" "2" "3" "4")
run_configs=("./solved/unconstrained_1-1temp.yml" "./solved/unconstrained_1-2.yml" "./solved/unconstrained_2.yml" "./solved/unconstrained_3.yml" "./solved/SCOSINE.yml")

timeout=3000
setup_name=tomislav_geno_oneatatime
results_directory="${timeout}/${setup_name}/"

for i in ${!run_names[*]}; do
    config=${run_configs[$i]}
    run_name=${run_names[$i]}
    nohup_file="nohup${run_name}.out"
    part_name="part${run_name}"
    result_directory="${results_directory}${part_name}"
    if [ -e "$nohup_file" ]; then
        echo $nohup_file already exists
        exit 1
    fi
    if [ -e "results/$result_directory" ]; then
        echo $result_directory already exists
        exit 1
    fi
    nohup python3 main.py --config ${config} --timeout ${timeout} --result_dir ${result_directory} > ${nohup_file}
done