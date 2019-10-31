import os
import sys
import pickle
from time import time
import numpy as np
import pandas as pd
import gzbuilder_analysis.parsing as pg
import gzbuilder_analysis.aggregation as ag
import gzbuilder_analysis.config as cfg
from gzbuilder_analysis.fitting import Model, fit_model, chisq
# from shapely.affinity import scale
# from descartes import PolygonPatch
from PIL import Image
import argparse

loc = os.path.abspath(os.path.dirname(__file__))
lib = os.path.join(loc, 'lib')
sid_list = np.loadtxt(
    os.path.join(lib, 'subject-id-list.csv'),
    dtype='u8'
)

parser = argparse.ArgumentParser(
    description=(
        'Fit Aggregate model and best individual'
        ' model for a galaxy builder subject'
    )
)
parser.add_argument('--index', '-i', metavar='N', default=-1, type=int,
                    help='Subject id index (from liv/subject-id-list.csv)')
parser.add_argument('--subject', '-s', metavar='N', default=-1, type=int,
                    help='Subject id')
parser.add_argument('--output', '-O', metavar='/path/to/output',
                    default='output_files/tuned_models',
                    help='Where to save the output tuned model')
parser.add_argument('--progress', '-P', action='store_true',
                    help='Whether to use a progress bar')
args = parser.parse_args()
if args.index == -1 and args.subject == -1:
    print('Invalid arguments')
    parser.print_help()
    sys.exit(0)
elif args.subject >= 0:
    subject_id = args.subject
else:
    subject_id = sid_list[args.index]

print(f'Working on {subject_id}')

im = np.array(Image.open('lib/subject_data/{}/image.png'.format(subject_id)))[::-1]
classifications = pd.read_csv('lib/galaxy-builder-classifications.csv', index_col=0)
fitting_metadata = pd.read_pickle('lib/fitting_metadata.pkl')
fm = fitting_metadata.loc[subject_id]
data = fm['galaxy_data']
sigma_image = fm['sigma_image']
psf = fm['psf']

gal = pd.read_csv('lib/gal-metadata.csv', index_col=0).loc[subject_id]

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

models = scaled_models.apply(
    pg.reproject_model,
    wcs_in=fm['montage_wcs'], wcs_out=fm['original_wcs']
)


def cluster_components(models, data, ba, angle):
    model_cluster = ag.cluster_components(
        models=models, image_size=data.shape,
        phi=angle, ba=ba,
    )
    aggregation_result = ag.aggregate_components(model_cluster)
    return model_cluster, aggregation_result


ba = gal['PETRO_BA90']
phi = np.rad2deg(gal['original_angle'])
print('Clustering all components in SDSS frame coordinates')
t0 = time()
clusters, agg_model_dict = cluster_components(
    models,
    fm['galaxy_data'],
    ba, phi,
)
print('\tElapsed time: {:.2f}'.format(time() - t0))

# save the results
cluster_output = os.path.join(lib, 'clusters')
os.makedirs(cluster_output, exist_ok=True)
with open(os.path.join(cluster_output, f'{subject_id}.pickle'), 'wb') as f:
    pickle.dump(clusters, f)

raw_agg_model_output_dir = os.path.join(lib, 'agg_models')
os.makedirs(raw_agg_model_output_dir, exist_ok=True)
raw_agg_model_output = os.path.join(raw_agg_model_output_dir, f'{subject_id}.json')
print(f'Writing aggregate model to {raw_agg_model_output}')
with open(raw_agg_model_output, 'w') as f:
    f.write(pg.make_json(agg_model_dict))


fm = fitting_metadata.loc[subject_id]
data = fm['galaxy_data']
sigma_image = fm['sigma_image']
psf = fm['psf']

# now to tune the model, this can take some time
print('Tuning model')
agg_model = Model(
    agg_model_dict,
    data,
    psf=psf,
    sigma_image=sigma_image
)
# tune the brightness to provide a good starting point
params_to_fit = {
    'disk':   ('I', 'Re'),
    'bulge':  ('I', 'Re'),
    'bar':    ('I', 'Re'),
    'spiral': ('I', 'spread', 'falloff'),
}
res, partially_tuned_model_dict = fg.fit_model(
    agg_model,
    params=params_to_fit,
    options=dict(maxiter=50)
)

# allow roll and position to vary
params_to_fit2 = {
    'disk':   ('mux', 'muy', 'roll', 'I', 'Re', 'q'),
    'bulge':  ('mux', 'muy', 'roll', 'I', 'Re', 'q', 'n'),
    'bar':    ('mux', 'muy', 'roll', 'I', 'Re', 'q', 'n', 'c'),
    'spiral': ('I', 'spread', 'falloff'),
}
res, tuned_model_dict = fg.fit_model(
    agg_model,
    params=params_to_fit2,
    options=dict(maxiter=500)
)


print(
    'Writing tuned aggregate model to',
    os.path.join(args.output, f'agg/{subject_id}.json')
)

with open(os.path.join(args.output, f'agg/{subject_id}.json'), 'w') as f:
    f.write(pg.make_json(tuned_model_dict))
