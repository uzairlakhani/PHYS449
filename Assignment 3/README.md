# PHYS449

## Dependencies

- json
- numpy
- matplotlib
- tensorflow
- seaborn

## Running `main.py`

To run `main.py`, use

```sh
python main.py --param param/param.json -v 2 --res-path plots
--x-field "-y/np.sqrt(x**2 + y**2)" --y-field "x/np.sqrt(x**2 + y**2)"
--lb -1.0 --ub 1.0 --n-tests 3

And here is an example run of argparse --help:
--------------------------------
usage: main.py [-h] [--param param.json] [-v N] [--res-path results]
[--x-field x**2] [--y-field y**2] [--lb LB] [--ub UB] [--n-tests N_TESTS]

ODE Solver

optional arguments:
  -h, --help          show this help message and exit
  --param param.json  file name for json attributes
  -v N                verbosity (default: 1)
  --res-path results  path to save the test plots at
  --x-field x**2      expression of the x-component of the vector field
  --y-field y**2      expression of the y-component of the vector field
  --lb LB             lower bound for initial conditions
  --ub UB             upper bound for initial conditions
  --n-tests N_TESTS   number of test trajectories to plot

```
