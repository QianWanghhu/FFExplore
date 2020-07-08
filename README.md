FFExplore
---------

Explores the effect of factor fixing on model outputs.

Companion code for the paper submitted to Reliability Engineering and System Safety. Code written by Qian Wang (@QianWanghhu) with input from Takuya Iwanaga (@ConnectedSystems) and released under the MIT license.

**To recreate results:**

1. Set up conda environment with `conda env create -f ffexplore.yml` 
        
    (Note that the environment spec is for Windows only. 
        Adjustments will be necessary for *nix platforms.)

    TODO: Create environment spec.

2. Clone and install other packages:

   2.1. Modified version of SALib package.

   Modifications include changes to return confidence intervals for parameter groupings.

    (TODO: Installation instructions)
   
   2.2 PAWN Package

   PAWN analyses were conducted with the package available for download after registration.

   https://www.safetoolbox.info/register-for-download/

4. Adjust output directory settings in `apply/settings.py`

5. Perform experiments:

   4.1. Run PAWN from the `apply/mp_pawn` directory, for example:

      `python run_mp_pawn.py --sample_range 1600 2000 --step 100 --tuning 2 4 --ncores 4 --fdir [output directory]`

   4.2. Run `apply/apply_morris.py` and `apply/apply_Sobol.py`

6. To recreate plots, run `apply/create_figures.py`

   Note that an Excel file will be created in the case of Figure 4.