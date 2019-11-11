import os
import re
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from gzbuilder_analysis.parsing import unmake_json
try:
    from gzbuilder_analysis.rendering.cuda import calculate_model
except ModuleNotFoundError:
    from gzbuilder_analysis.rendering import calculate_model
from gzbuilder_analysis.fitting import chisq
from lib.galaxy_utilities import get_psf


loc = os.path.abspath(os.path.dirname(__file__))
lib = os.path.join(loc, 'lib')

parser = argparse.ArgumentParser(
    description=(
        'Fit Aggregate model and best individual'
        ' model for a galaxy builder subject'
    )
)
parser.add_argument('--tuned_input', metavar='/path/to/tuned_models',
                    default=os.path.join(loc, 'output_files/tuned_models'),
                    help='Location of tuned models')
parser.add_argument('--aggregate-models', metavar='/path/to/agg_models',
                    default=os.path.join(lib, 'agg_models'),
                    help='Location of aggregate models')
parser.add_argument('--volunteer-models', metavar='/path/to/volunteer_models',
                    default=os.path.join(lib, 'volunteer_models'),
                    help='Location of volunteer models')
parser.add_argument('--output', metavar='/path/to/output.pickle',
                    default=os.path.join(lib, 'models.pickle'),
                    help='Where to save the output tuned model')

args = parser.parse_args()


fitting_metadata = pd.read_pickle('lib/fitting_metadata.pkl')


def filter_model_file(f):
    return bool(re.match(r'[0-9]+\.(?:json|pickle)', f))


def get_subject_id(f):
    try:
        return int(re.search(r'([0-9]+)\.(?:json|pickle)', f).group(1))
    except ValueError:
        return np.nan


def load_model_file(loc):
    with open(loc) as f:
        model = unmake_json(f)
    return model


def get_best_model(f):
    df = pd.read_pickle(f)
    return df.model[df.chisq.idxmin()], df.chisq.min()


def get_chisq(row):
    fm = fitting_metadata.loc[row.subject_id]
    data = fm.galaxy_data
    sigma_image = fm.sigma_image
    psf = get_psf(row.subject_id)
    return chisq(calculate_model(row.model, data.shape, psf), data, sigma_image)


tuned_agg_models_dir = os.path.join(args.tuned_input, 'agg')
tuned_bi_models_dir = os.path.join(args.tuned_input, 'bi')

agg_models = pd.Series({
    get_subject_id(f): load_model_file(os.path.join(args.aggregate_models, f))
    for f in os.listdir(args.aggregate_models)
    if filter_model_file(f)
}, name='aggregate')


bi_models = pd.DataFrame.from_dict(
    {
        get_subject_id(f): get_best_model(os.path.join(args.volunteer_models, f))
        for f in os.listdir(args.volunteer_models)
        if filter_model_file(f)
    },
    orient='index',
    columns=('best_individual', 'best_individual_chisq')
)

tuned_agg_models = pd.Series({
    get_subject_id(f): load_model_file(os.path.join(tuned_agg_models_dir, f))
    for f in os.listdir(tuned_agg_models_dir)
    if filter_model_file(f)
}, name='tuned_aggregate')

tuned_bi_models = pd.Series({
    get_subject_id(f): load_model_file(os.path.join(tuned_bi_models_dir, f))
    for f in os.listdir(tuned_bi_models_dir)
    if filter_model_file(f)
}, name='tuned_best_individual')

# now to calucate chisq values for tuned models:
tqdm.pandas(desc='Calculating chisq (tuned aggregate)', leave=False)
tuned_agg_chisq = tuned_agg_models\
    .rename('model')\
    .reset_index()\
    .rename(columns=dict(index='subject_id'))\
    .set_index('subject_id', drop=False)\
    .progress_apply(get_chisq, axis=1)\
    .rename('tuned_aggregate_chisq')

tqdm.pandas(desc='Calculating chisq (tuned best individual)', leave=False)
tuned_bi_chisq = tuned_bi_models\
    .rename('model')\
    .reset_index()\
    .rename(columns=dict(index='subject_id'))\
    .set_index('subject_id', drop=False)\
    .progress_apply(get_chisq, axis=1)\
    .rename('tuned_best_individual_chisq')

df = pd.concat((
    agg_models, bi_models,
    tuned_agg_models, tuned_bi_models,
    tuned_agg_chisq, tuned_bi_chisq
), axis=1)

df.to_pickle(args.output)
