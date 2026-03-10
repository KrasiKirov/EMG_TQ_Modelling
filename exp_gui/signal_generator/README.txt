Directory Description:
This directory contains the GUIs and functions responsible for generating large amplitude position signals and perturbation sequences.

GUIs:

I) Ali_exp_gui.m & Ali_exp_gui.fig
The experiment control GUI was created for SPASTIC I (2019) using GUIDE (currently using MATLAB App Designer).
Two other GUIs (Signal_generator.m and Perturbation_generator.m) are called from inside Ali_exp_gui by clicking "New" in the dropdown menus.

II) Signal_generator.m & Signal_generator.fig
Allows the user to create large amplitude signals with desired parameters and save the signal in a .mat file.

III) Perturbation_generator.m
Allows the user to create realizations of PRALDS and PRBS with desired parameters and save the signal in a .mat file.

The signals are saved as structures. Each structure has a field called Sig which contains a 2xlength matrix of time vector and position values. The structures also have fields for each parameter used to generate the input. The destination folder for saving the files must be hardcoded in the GUI callback function for the SAVE button; otherwise, it will cause errors.

Functions:

I) add_transitions.m
Uses the gen_ramp_and_hold.m function to concatenate input with ramp signals at the beginning and end of the signal to prevent sudden jumps at the beginning and end of experiments.


II) gen_[signal type].m
These functions create large amplitude position signals with desired parameters.