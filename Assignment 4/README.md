# PHYS449

## Dependencies

- json
- numpy
- matplotlib
- torch
- seaborn

## Running `main.py`

To run `main.py`, use

```sh
python main.py --param param/param.json --data-path data/in.txt --res-path results -v 1

And here is an example run of argparse --help:
--------------------------------
usage: main.py [-h] [--param param.json] [--data-path data] [--res-path results] [-v N]

Fully Visible Boltzmann Machine

optional arguments:
  -h, --help          show this help message and exit
  --param param.json  filename for json atributes
  --data-path data    path to get the data from
  --res-path results  path to save the plot and output at
  -v N                verbosity (default: 1). When verbosity is 1, KL Divergence shown.
```
