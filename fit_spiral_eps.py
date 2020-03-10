import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.cluster import DBSCAN
import pandas as pd
from astropy.io import fits
import gzbuilder_analysis.parsing as pg
import gzbuilder_analysis.aggregation as ag
import gzbuilder_analysis.aggregation.jaccard as jaccard
from gzbuilder_analysis.config import ARM_CLUSTERING_PARAMS
from PIL import Image
import lib.galaxy_utilities as gu

BAR_MIN_SAMPLES = 5

try:
    loc = os.path.abspath(os.path.dirname(__file__))
except NameError:
    loc = '.'
lib = os.path.join(loc, 'lib')


def get_n_arms(gal):
    keys = (
        't11_arms_number_a31_1_debiased',
        't11_arms_number_a32_2_debiased',
        't11_arms_number_a33_3_debiased',
        't11_arms_number_a34_4_debiased',
        't11_arms_number_a36_more_than_4_debiased',
    )
    return sum((i + 1) * gal[k] for i, k in enumerate(keys))


NSA_GZ = fits.open('../source_files/NSA_GalaxyZoo.fits')
classifications = pd.read_csv('lib/galaxy-builder-classifications.csv',
                              index_col=0)
fitting_metadata = pd.read_pickle('lib/fitting_metadata.pkl')
gal_df = pd.read_csv('lib/gal-metadata.csv', index_col=0)

gz2_n_arms = {}
spiral_distances = {}

with tqdm(fitting_metadata.index) as pbar:
    for subject_id in pbar:
        pbar.set_description(f'Subject: {subject_id}')
        metadata = gu.metadata.loc[subject_id]
        gal = NSA_GZ[1].data[
            NSA_GZ[1].data['dr7objid'] == np.int64(metadata['SDSS dr7 id'])
        ]
        if len(gal) > 0:
            gz2_n = get_n_arms(gal[0])
            if gz2_n > 3:
                # ignore galaxies with too many arms (bad practise)
                continue
            gz2_n_arms[subject_id] = get_n_arms(gal[0])
            # this is a faster (and identical) way to calculate distance
            # matrices
            agg_res = pd.read_pickle(
                f'output_files/aggregation_results/{subject_id}.pkl.gz'
            )
            spiral_distances[subject_id] = agg_res.spiral_pipeline.distances


def func(log10_eps, SPIRAL_MIN_SAMPLES):
    db = DBSCAN(
        eps=10**log10_eps,
        min_samples=SPIRAL_MIN_SAMPLES,
        metric='precomputed',
        n_jobs=-1,
        algorithm='brute',
    )
    n_correct = 0
    for subject_id in fitting_metadata.index:
        if subject_id not in gz2_n_arms or gz2_n_arms[subject_id] > 3:
            continue
        n_arms = db.fit(spiral_distances[subject_id]).labels_.max() + 1
        correct = round(gz2_n_arms[subject_id], 0) == n_arms
        n_correct += int(round(gz2_n_arms[subject_id], 0) == n_arms)
    return -n_correct + 10**log10_eps / 100

res = {}
for i in range(3, 8):
    r = minimize_scalar(func, args=(i,), method='Bounded', bounds=(-5, 1))
    if r['success']:
        res[i] = r
        print('Fun: {:.4f}: min_samples={} eps={:.4f}'.format(
            r['fun'], i, 10**r['x']
        ))

pd.to_pickle(res, 'bar_eps_fit_result.pkl.gz')
