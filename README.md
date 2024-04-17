# ML Benchmarks
Run
```console
python3 main.py --help
```
to see available optimization problems and solvers.

To get solve problems and get results, set the string *config* to the yaml config file where you listed all desired optimization problems and solvers and run
```console
python3 main.py
```
You can benchmark multiple optimization problems and solvers simultaneously. Plots and results are saved in `./figures/`.


To build the f, g, and H timing solver on your machine, download go to https://portal.ampl.com/~dmg/netlib/ampl/ download solvers2.tgz and build fgh_timing_solver.c according to the instructions. You can also try using the built fgh_timing_solver, but I don't think it will work on any other machine.

You also need to install AMPL on your system, but that's reasonably simple to do.
