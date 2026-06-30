# MARACAS
:warning: Work in progress (1D is not ready yet) :warning:
## Installation
To run maracas, you'll need `python`, `numpy`, `numba`, `pytables`, `jsonc`, `scipy`, `pytables`, `pyvista`, `matplotlib` and `colorcet`.

## Configure your simulation
Use, or update, the input json files given in the `inputs/` repository.</br>
All units are in meters, volt and second.</br>
Drift is along the *x axis*.

## Run your simulation
The command to run maracas is:</br>
`python maracas.py -i path_to_input_file.json -o output_name`

An `hdf5` file will be produced, containing the charge density and field maps, as well as the forward and backward distortion maps.</br>
Forward distortions: to go from true to reco space (`reco = true - f_distortion`)</br>
Backward distortions: to go from reco to true space (`true = reco - b_distortion`)</br>
