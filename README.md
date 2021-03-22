# reconstruction-metrics

#### Calculates the performance metrics of event reconstruction methods for the SiFi-CC.


To install the required packages:

`pip install uproot==3.13.0 numpy awkward lz4 xxhash`

Usage

```
python evaluate.py [-h] -f FILE [-d DIRECTORY] [--e_pos_x VALUE] [--e_pos_y VALUE] [--e_pos_z VALUE] [--p_pos_x VALUE] [--p_pos_y VALUE] [--p_pos_z VALUE] [--e_energy VALUE] [--p_energy VALUE]

required arguments:
  -f FILE, --file FILE  The root file containing the reconstructed events

optional arguments:
  -h, --help            show the help message and exit
  -d DIRECTORY, --source_dir DIRECTORY
                        Directory of the source simulation root file. Default directory is ./
  --e_pos_x VALUE       The distance limit for the x-axis of the electon. Default is 2.6 mm
  --e_pos_y VALUE       The distance limit for the y-axis of the electon. Default is 10 mm
  --e_pos_z VALUE       The distance limit for the z-axis of the electon. Default is 2.6 mm
  --p_pos_x VALUE       The distance limit for the x-axis of the photon. Default is 2.6 mm
  --p_pos_y VALUE       The distance limit for the y-axis of the photon. Default is 10 mm
  --p_pos_z VALUE       The distance limit for the z-axis of the photon. Default is 2.6 mm
  --e_energy VALUE      The energy difference limit relative to the electron energy. Default is .12
  --p_energy VALUE      The energy difference limit relative to the photon energy. Default is .12
```
