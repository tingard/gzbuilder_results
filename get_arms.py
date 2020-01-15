import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

loc = os.path.abspath(os.path.dirname(__file__))
lib = os.path.join(loc, 'lib')

parser = argparse.ArgumentParser(
    description=(
        'Fit Aggregate model and best individual'
        ' model for a galaxy builder subject'
    )
)
parser.add_argument('--clusters', metavar='/path/to/clusters',
                    default=os.path.join(lib, 'clusters'),
                    help='Location of tuned models')
parser.add_argument('--output', metavar='/path/to/output.pickle',
                    default=os.path.join('spiral_arms.pickle'),
                    help='Location of tuned models')
parser.add_argument(
    '--no-weights',
    action='store_false',
    help='Whether to weight points by R^2 and number of arms present')

args = parser.parse_args()

def filter_model_file(f):
    return bool(re.match(r'[0-9]+\.(?:json|pickle)', f))


def get_subject_id(f):
    try:
        return int(re.search(r'([0-9]+)\.(?:json|pickle)', f).group(1))
    except ValueError:
        return np.nan


def get_arms(f):
    df = pd.read_pickle(f)
    pipeline = df['spiral']
    arms = pipeline.get_arms(weight_points=True)
    return dict(
        pipeline=pipeline,
        **{f'arm_{i}': arm for i, arm in enumerate(arms)}
    )


files = pd.Series(os.listdir(args.clusters))
files = files.where(files.apply(filter_model_file)).dropna()
tqdm.pandas(desc='Getting spiral arms')
arms = files.progress_apply(lambda f: get_arms(os.path.join(args.clusters, f)))
arms.index = files.apply(get_subject_id).values
arms = arms.apply(pd.Series)
arms.to_pickle(args.output)
# arms = pd.DataFrame.from_dict(
#     {
#         get_subject_id(f): get_arms(os.path.join(args.clusters, f))
#         for f in os.listdir(args.clusters)[:3]
#         if filter_model_file(f)
#     },
#     orient='index',
#     columns=('best_individual', 'best_individual_chisq')
# )
