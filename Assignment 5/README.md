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
python main.py --param param/param.json -o result_dir -n 100 -v 1

And here is an example run of argparse --help:
--------------------------------
usage: main.py [-h] [--param param.json] [-o results] [-n N_Images] [-v N]

Variational Auto-Encoder

optional arguments:
  -h, --help          show this help message and exit
  --param param.json  filename for json atributes
  -o results          directory where the results are saved
  -n N_Images         number of digit sample images
  -v N                verbosity (default: 1). When verbosity is 1, Loss is shown.

```
