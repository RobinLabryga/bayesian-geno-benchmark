# Bayesian Geno Benchmarks

This is the repo used for the experiments of the paper "Information Preserving Line Search via Bayesian Optimization".

The instructions to reproduce the results are below.
We have not tested if this works flawlessly on different machines, and the instructions are from memory so they might be a bit off.
Feel free to open an issue or pull request if you have any suggestions/questions.

## Instructions

1. When cloning the repository, do so recursively to get the submodules
1. Install CUTEst as per the instruction in the CUTEst submodule
1. Set the required environment variables as

    ```bash
        export ARCHDEFS="$PWD"/CUTEst/ARCHDefs/
        export SIFDECODE="$PWD"/CUTEst/SIFDecode/
        export MASTSIF="$PWD"/CUTEst/sif/
        export CUTEST="$PWD"/CUTEst/CUTEst/
        export MYARCH="pc64.lnx.gfo" or export MYARCH="mac64.osx.gfo"
    ```

1. See the available option for the benchmar using

    ```bash
        python3 main.py --help
    ```

1. To see a summary of the resulting run use

    ```bash
        python3 show_dataframe.py -f ./results/res1/results.pkl ./results/res2/results.pkl --plot_again --single_fig_per_f_over_time --barplots_cnt 10
    ```

1. The shell script used to run the all the problems for a specific solver is `run_problems.sh`.
1. The config files used for the experiments are in `./solved/`.
