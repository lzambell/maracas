# MARACAS
:warning: Work in progress (1D and 3D not ready yet) :warning:
## Installation
To run maracas, you'll need `python`, `numpy`, `numba`, `pytables` and `matplotlib`.

## Configure your simulation
Use, or update, the input json files given in the `input/` repository.</br>
All units are in meters, volt and second.

## Run your simulation
The command to run maracas is:</br>
`python maracas.py -i path_to_input_file.json -o output_name`

It will generate the charge density and field maps once the equations have converged as well as the distortion maps (as in true-reconstructed) in a `hdf5`file.