#!/bin/bash

export ARCHDEFS="$PWD"/CUTEst/ARCHDefs/
export SIFDECODE="$PWD"/CUTEst/SIFDecode/
export MASTSIF="$PWD"/CUTEst/sif/
export CUTEST="$PWD"/CUTEst/CUTEst/
export MYARCH="pc64.lnx.gfo"

run_names=("4" "1-1" "1-2" "2" "3-1" "3-2" "bound_0" "bound_1" "bound_2" "bound_3" "bound_4" "bound_5")
run_configs=("./solved/SCOSINE.yml" "./solved/unconstrained_1-1.yml" "./solved/unconstrained_1-2.yml" "./solved/unconstrained_2.yml" "./solved/unconstrained_3-1.yml" "./solved/unconstrained_3-2.yml" "./solved/bound_0.yml" "./solved/bound_1.yml" "./solved/bound_2.yml" "./solved/bound_3.yml" "./solved/bound_4.yml" "./solved/bound_5.yml")

timeout=3000
setup_name=scipy
solver_config="solved/ScipySolver.yml"
results_directory="${timeout}/${setup_name}/"

for i in ${!run_names[*]}; do
    config=${run_configs[$i]}
    run_name=${run_names[$i]}
    part_name="part${run_name}"
    result_directory="${results_directory}${part_name}"
    if [ -e "results/$result_directory" ]; then
        echo $result_directory already exists
        continue
    fi
    python3 -u main.py --problem_config ${config} --solver_config ${solver_config} --timeout ${timeout} --result_dir ${result_directory}
done