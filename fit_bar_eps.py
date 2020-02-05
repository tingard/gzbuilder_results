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
from PIL import Image
import lib.galaxy_utilities as gu

BAR_MIN_SAMPLES = 5

loc = os.path.abspath(os.path.dirname(__file__))
lib = os.path.join(loc, 'lib')


def get_pbar(gal):
    n = gal['t03_bar_a06_bar_debiased'] + gal['t03_bar_a07_no_bar_debiased']
    return gal['t03_bar_a06_bar_debiased'] / n


NSA_GZ = fits.open('../source_files/NSA_GalaxyZoo.fits')
classifications = pd.read_csv('lib/galaxy-builder-classifications.csv',
                              index_col=0)
fitting_metadata = pd.read_pickle('lib/fitting_metadata.pkl')
gal_df = pd.read_csv('lib/gal-metadata.csv', index_col=0)

geom_dict = {}
gz2_pbar = {}
bar_distances = {}

with tqdm(fitting_metadata.index) as pbar:
    for subject_id in pbar:
        pbar.set_description(f'Subject: {subject_id}')
        im = np.array(Image.open('lib/subject_data/{}/image.png'.format(
            subject_id
        )))[::-1]
        fm = fitting_metadata.loc[subject_id]
        data = fm['galaxy_data']
        # gal = gal_df.loc[subject_id]
        metadata = gu.metadata.loc[subject_id]
        gal = NSA_GZ[1].data[
            NSA_GZ[1].data['dr7objid'] == np.int64(metadata['SDSS dr7 id'])
        ]
        gz2_pbar[subject_id] = get_pbar(gal[0]) if len(gal) > 0 else np.nan
        c = classifications.query('subject_ids == {}'.format(subject_id))
        zoo_models = c.apply(
            pg.parse_classification,
            axis=1,
            image_size=np.array(im.shape),
            size_diff=im.shape[0] / data.shape[0],
            ignore_scale=True  # ignore scale slider when aggregating
        )
        scaled_models = zoo_models.apply(
            pg.scale_model,
            args=(fm['size_diff'],),
        )
        rotated_models = scaled_models.apply(
            pg.rotate_model_about_centre,
            args=(
                np.array(im.shape) * fm['size_diff'],
                fm.rotation_correction
            ),
        )
        models = scaled_models.apply(
            pg.reproject_model,
            wcs_in=fm['montage_wcs'],
            wcs_out=fm['original_wcs']
        )
        geoms = models.apply(ag.get_geoms).apply(pd.Series)
        geom_dict[subject_id] = geoms
        bar_distances[subject_id] = jaccard.make_jaccard_distances(
            geoms['bar'].dropna()
        )

comp = pd.Series(gz2_pbar)
best_score = (
    (comp > 0.5).sum()
    + (comp < 0.2).sum()
)


def func(eps, BAR_MIN_SAMPLES):
    eps = max(1E-7, eps)
    n_correct = 0
    for subject_id in fitting_metadata.index:
        distances = bar_distances[subject_id]
        clf = DBSCAN(eps=eps, min_samples=BAR_MIN_SAMPLES,
                     metric='precomputed')
        clf.fit(distances)
        if subject_id not in gz2_pbar:
            pass
        elif gz2_pbar[subject_id] < 0.2 and np.max(clf.labels_) < 0:
            # gz2 says unlikely to have a bar, and we have no bar
            n_correct += 1
        elif gz2_pbar[subject_id] > 0.5 and np.max(clf.labels_) >= 0:
            # gz2 says likely to have a bar, and we have a bar
            n_correct += 1
    return -n_correct


res = {}
for i in range(3, 8):
    r = minimize_scalar(func, args=(i,), method='Bounded', bounds=(1E-7, 2))
    if r['success']:
        res[i] = r
        print('Fun: {} / {}: min_samples={} eps={}'.format(
            -r['fun'], best_score, i, r['x']
        ))

pd.to_pickle(res, 'bar_eps_fit_result.pkl.gz')
