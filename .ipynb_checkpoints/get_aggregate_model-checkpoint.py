import os
from os.path import join
from time import sleep
from tqdm import tqdm
import numpy as np
import pandas as pd
import gzbuilder_analysis.parsing as pg
import gzbuilder_analysis.aggregation as ag
# from shapely.affinity import scale
# from descartes import PolygonPatch
from PIL import Image
import argparse

loc = os.path.abspath(os.path.dirname(__file__))
lib = os.path.join(loc, 'lib')

parser = argparse.ArgumentParser(
    description=(
        'Calculate aggregation results for galaxy builder subjects'
    )
)
parser.add_argument('--output', '-O', metavar='/path/to/output',
                    default=join(loc, 'output_files/aggregation_results'),
                    help='Where to save the output tuned model')
parser.add_argument('--subjects', metavar='subject_ids', type=int, nargs='+',
                    help='Subject ids to work on (otherwise will run all)')
args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)
classifications = pd.read_csv('lib/galaxy-builder-classifications.csv', index_col=0)
classifications.created_at = pd.to_datetime(classifications.created_at)
fitting_metadata = pd.read_pickle('lib/fitting_metadata.pkl')
gal_df = pd.read_csv('lib/gal-metadata.csv', index_col=0)

with tqdm(args.subjects or fitting_metadata.index) as pbar:
    for subject_id in pbar:
        pbar.set_description('Subject: {}'.format(subject_id))
        im = np.array(Image.open('lib/subject_data/{}/image.png'.format(subject_id)))[::-1]
        fm = fitting_metadata.loc[subject_id]
        data = fm['galaxy_data']
        gal = gal_df.loc[subject_id]
        # take the first 30 classifications recieved for this galaxy
        c = (classifications
            .query('subject_ids == {}'.format(subject_id))
            .sort_values(by='created_at')
            .head(30)
        )
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
        models = rotated_models.apply(
            pg.reproject_model,
            wcs_in=fm['montage_wcs'], wcs_out=fm['original_wcs']
        )
        sanitized_models = models.apply(pg.sanitize_model)
        try:
            aggregation_result = ag.AggregationResult(sanitized_models, data)
            pd.to_pickle(
                aggregation_result,
                join(args.output, '{}.pkl.gz'.format(subject_id))
            )
        except TypeError:
            print('No disk cluster for {}'.format(subject_id))
