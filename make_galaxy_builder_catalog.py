# TODO: correct i0 to ie and rEff to Re (easy?)
# TODO: should do any corrections to c? (hard)

import json
import numpy as np
import pandas as pd
from gzbuilder_analysis import parsing
import lib.galaxy_utilities as gu
import tqdm as tqdm
from datetime import datetime


# Configuration
# What keys should be grabbed from the galaxy metadata?
METADATA_KEYS = ['ra', 'dec', 'Run', 'Field', 'Rerun', 'Camcol',
                 'NSA id', 'redshift', 'SDSS dr7 id']
RENAME_MAP = {
    'ra': 'Ra', 'dec': 'Dec', 'NSA id': 'NSAID', 'redshift': 'z',
    'SDSS dr7 id': 'DR7OBJID',
}

# Loading in required files
sid_list_values = np.loadtxt('lib/subject-id-list.csv', dtype='u8')
aggregation_results = pd.read_pickle('lib/aggregation_results.pickle')
best_models = pd.read_pickle('lib/best_individual.pickle')
fitted_models = pd.read_pickle('lib/fitted_models.pickle')


# Preprocessing and feature extraction
sid_list = pd.Series(
    sid_list_values,
    index=sid_list_values
).rename('subject_id')

classifications = gu.classifications[
    np.isin(gu.classifications.subject_ids, sid_list)
]
annotations = classifications.annotations.apply(json.loads)

# Get metadata that was bundled with Zooniverse Subjects
metadata = pd.concat(
    (sid_list.apply(gu.meta_map.get).apply(pd.Series), sid_list),
    axis=1
)[METADATA_KEYS]

# Get rendering data provided to Zooniverse rendering code
diff_data = sid_list.apply(gu.get_diff_data).apply(pd.Series)
size_diffs = diff_data.eval('width / imageWidth')


def make_all_model_df(output=None):
    """Produces a CSV containing all models created by volunteers, scaled to
    Sloan pixels and in units of nanomaggies
    """
    # need to match index of models (which is classification ID)
    model_metadata = metadata.reindex(
        classifications.subject_ids
    ).set_index(annotations.index)
    models = pd.Series([
        parsing.parse_annotation(*i)
        for i in zip(
            annotations.values,
            size_diffs.reindex(classifications.subject_ids.values)
        )
    ], index=classifications.index).apply(parsing.make_json)

    rendering_multipliers = diff_data.multiplier.reindex(
        classifications.subject_ids
    )
    rendering_multipliers.index = classifications.index
    unscaled_models = pd.Series([
        parsing.make_unscaled_model(model, multiplier)
        for model, multiplier in zip(models, rendering_multipliers)
    ], index=models.index)

    models_df = pd.concat((
        unscaled_models.apply(json.dumps).rename('model'),
        model_metadata.rename(RENAME_MAP, axis=1),
    ), axis=1)

    # we don't want to include the classification ID in the CSV, so drop
    # the index
    output = (
        output if output is not None
        else 'GZB_ALL_MODEL_CATALOG_{}.csv'.format(
            datetime.now().strftime('%d-%m-%Y')
        )
    )
    models_df.to_csv(output, index=False)


def make_compiled_model_df(output=None):
    """Produces a CSV containing tuned aggregate, best individual and tuned
    best individual models, scaled to Sloan pixels and in units of nanomaggies.
    Includes MSE loss for tuned best individual and tuned aggregate models.
    """
    aggregate_errors = aggregation_results.Errors.apply(parsing.make_json)\
        .apply(json.dumps).reindex(sid_list.index).rename('aggregate_errors')

    def get_spiral_info(arms):
        if len(arms) > 0:
            return pd.Series(
                [*arms[0].get_parent().get_pitch_angle(arms), len(arms)],
                index=('pa', 'sigma_pa', 'n_spiral_arms')
            )
        return pd.Series(dict(pa=np.nan, sigma_pa=np.nan, n_spiral_arms=0))

    spiral_info = aggregation_results.Arms.apply(get_spiral_info)
    models_df = pd.concat((
        best_models['Model'].apply(parsing.make_json).apply(json.dumps)
            .reindex(sid_list.index).rename('best_individual_model'),
        fitted_models['agg'].apply(parsing.make_json).apply(json.dumps)
            .reindex(sid_list.index).rename('aggregate_model'),
        aggregate_errors,
        fitted_models.agg_loss.reindex(sid_list.index)
            .rename('tuned_aggregate_mse'),
        fitted_models.bi_loss.reindex(sid_list.index)
            .rename('tuned_best_individual_mse'),
        spiral_info.reindex(sid_list.index),
        metadata.reindex(sid_list.index).rename(RENAME_MAP, axis=1),
    ), axis=1)
    output = (
        output if output is not None
        else 'GZB_COMPILED_MODEL_CATALOG_{}.csv'.format(
            datetime.now().strftime('%d-%m-%Y')
        )
    )
    models_df.to_csv(output, index=False)


def make_rendering_data_df(output=None):
    data = (diff_data['imageData'] * diff_data['multiplier'])\
        .apply(lambda a: a[::-1]).rename('data')
    masks = 1 - diff_data['mask'].apply(lambda a: np.array(a)[::-1])\
        .rename('mask')
    psfs = diff_data['psf']
    make_list = lambda l: l.tolist()
    data_df = pd.concat((
        data.reindex(sid_list.index).apply(make_list).apply(json.dumps),
        masks.reindex(sid_list.index).apply(make_list).apply(json.dumps),
        psfs.reindex(sid_list.index).apply(make_list).apply(json.dumps),
        metadata.reindex(sid_list.index).rename(RENAME_MAP, axis=1),
    ), axis=1)
    output = (
        output if output is not None
        else 'GZB_DATA_CATALOG_{}.csv'.format(
            datetime.now().strftime('%d-%m-%Y')
        )
    )
    data_df.to_csv(output, index=False)

if __name__ == '__main__':
    # make_all_model_df()
    # make_compiled_model_df()
    make_rendering_data_df()
