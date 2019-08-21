# Galaxy builder aggregation

This is a collection of scripts and jupyter notebooks used to perform data reduction and aggregation on the output classifications of Galaxy Builder.

## Getting started

In order to run analysis, you will need to run a few data files:


- `lib/galaxy-builder-subjects`, a Galaxy Builder Zooniverse Subjects export
- `lib/galaxy-builder-classifications`, a Galaxy Builder Zoonvierse Classifications export
- `/lib/gal-metadata.csv`, which can be created using the `lib/make_galaxy_angle_dataframe.py` script (see python file for more instructions)
- `/lib/subject_id_list.csv`, a newline-separated file containing zooniverse subject ids of interest. Can be quickly created from the subject export using pandas.
- `lib/fitting_metadata.pkl`
	- obtained using `lib/make_fitting_metadata.py` and the files found in `/lib/subject_data/[SUBJECT_IDS]/[SUBJECT_DATA_FILES]`, which in turn can be created using `/lib/get_zooniverse_files.py`, ideally using the original source files for the galaxy builder subjects (created using [gzbuilder\_data\_prep](https://github.com/tingard/gzbuilder_data_prep)), or can be downloaded from the zooniverse api (which is untested)

Feel free to message the repository owner for tarballs of required files.

## Obtaining models

Galaxy Builder identifies two methods of obtaining a photometric model for a galaxy - either through clustering and aggregation of volunteer models or obtaining the best individual model for each galaxy. Each of these models is then tuned using a simple gradient-descent optimizer (L\_BFGS-b algorithm, implemented in Scipy) to finalize model parameters.

In order to calculate the best individual and aggregate models, run `/input_files/make_agg_bi_models.ipynb`. Note this will probably take ~1.5 hours so we recommend running this on an HPC using the `papermill` package.

This notebook will create `/lib/aggregation_results.pickle` and `/lib/best_individual.pickle` (unless other locations are provided using papermill by overwriting the `AGGREGATE_LOCATION` and/or `BEST_INDIVIDUAL_LOCATION` parameters).

`/lib/aggregation_results.pickle` contains a pandas DataFrame indexed on Zooniverse subject id, with columns **Model**, **Errors**, **Masks** and **Arms**. **Model** contains the JSON model dictionary for rendering, **Errors** contains the sample variance in measured parameters for shapes clustered for each component, **Masks** contains a mask for each component indicating which classifications were determined to be in the cluster (excluding spiral arms). **Arms** contains `gzbuilder.spirals.oo.Arm` objects for the spiral arms present in the aggregate model, which contain information on clustered drawn poly-lines, point cleaning and derived values such as pitch angle and length.

Once these files have been created, model tuning can be done using the `/tune_model.py` script, more info on which can be found in the python file. Again we recommend running this in batch mode on an HPC, an example Slurm array job can be found in `input_files/job__tune_models.job`. This will create many JSON files in `/output_files/tuned_models`, which can be compiled into a catalogue using the `/input_files/compile_tuned_models.ipynb` notebook.

The `tune_model.py` script makes use of `gzbuilder_analysis.fitting` to instantiate `Model` objects, allowing efficient rendering of galaxy models. If you wish to perform your own optimization, we suggest using `gzbuilder_analysis.fitting` either directly as a reference, as it is non-trivial to compare galaxy builder models to the data, due to a couple of poor choices during the Zooniverse project creation process.

A bash script to perform model calculation, tuning and compilation on a Slurm-based HPC therefore looks something like this:

```
NSID=$(expr $(cat lib/subject-id-list.csv | wc -l) - 1)

srun papermill input_files/make_agg_bi_models.ipynb \
               output_files/make_agg_bi_models.ipynb

tuning_jid=$(sbatch --array=0-$NSID input_files/job__tune_models.job)

srun papermill input_files/compile_tuned_models.ipynb \
               output_files/compile_tuned_models.ipynb \
               --dependency=afterany:$tuning_jid
```

Where `lib/subject-id-list.csv` is the newline-separated list of zooniverse subject ids to work on.
## Performing analysis

Once models have been obtained
