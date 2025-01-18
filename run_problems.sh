#!/bin/bash

export ARCHDEFS="$PWD"/CUTEst/ARCHDefs/
export SIFDECODE="$PWD"/CUTEst/SIFDecode/
export MASTSIF="$PWD"/CUTEst/sif/
export CUTEST="$PWD"/CUTEst/CUTEst/
export MYARCH="pc64.lnx.gfo"

run_names=(
    "unconstrained_0"
    "unconstrained_1"
    "unconstrained_2"
    "unconstrained_3"
    "unconstrained_4"
    "unconstrained_5"
    "unconstrained_6"
    "unconstrained_7"
    "unconstrained_8"
    "unconstrained_9"
    "bound_0"
    "bound_1"
    "bound_2"
    "bound_3"
    "bound_4"
    "bound_5"
)

timeout=3000
setup_name=scipy
solver_config="solved/ScipySolver.yml"
results_directory="${timeout}/${setup_name}/"

for i in ${!run_names[*]}; do
    run_name=${run_names[$i]}
    problem_config="./solved/${run_name}.yml"
    result_directory="${results_directory}${run_name}"
    if [ -e "results/$result_directory" ]; then
        echo $result_directory already exists
        continue
    fi
    python3 -u main.py --problem_config ${problem_config} --solver_config ${solver_config} --timeout ${timeout} --result_dir ${result_directory}
done
