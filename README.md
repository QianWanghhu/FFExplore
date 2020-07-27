FFExplore
---------

Explores the effect of factor fixing on model outputs.

Companion code for the paper submitted to Reliability Engineering and System Safety. Code written by Qian Wang (@QianWanghhu) with input from Takuya Iwanaga (@ConnectedSystems) and released under the MIT license.

**To recreate results:**

1. Set up conda environment with `conda env create -f ffexplore.yml` 
        
    (Note that the environment spec is for Windows only. 
        Adjustments will be necessary for *nix platforms.)

2. Clone and install other packages:

   2.1. Modified version of SALib package.

      Modifications include changes to return confidence intervals for parameter rankings.

      The branch named as product_dist with corresponding modifications can be downloaded or forked following the link below:
      https://github.com/QianWanghhu/SALib/tree/product_dist 

      `git clone --single-branch --branch product_dist https://github.com/QianWanghhu/SALib salib-prod`

      The above will clone the repository into a directory `salib-prod`.

      Navigate to the directory and install with:

      `pip install .`

3. Adjust output directory settings in `apply/settings.py`

   Directory locations should be specified relative to the project root directory.

   Don't forget to create the specified folders.

4. Perform experiments:

   All instructions here assume the code is being run from the project root directory.

   Run `apply/apply_Morris.py` and `apply/apply_Sobol.py`

5. To recreate plots, run `apply/create_figures.py`

   Note that an Excel file will be created in the case of Figure 4 and further polishing work may be needed to achieve the Figure 4 as the one in the paper.