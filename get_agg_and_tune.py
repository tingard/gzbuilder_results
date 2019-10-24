import os
import sys
import pickle
from time import time
import numpy as np
import pandas as pd
import gzbuilder_analysis.parsing as pg
import gzbuilder_analysis.aggregation as ag
import gzbuilder_analysis.config as cfg
from gzbuilder_analysis.fitting.model import Model, fit_model
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
fm = pd.read_pickle('lib/fitting_metadata.pkl').loc[subject_id]
data = fm['galaxy_data']
sigma = fm['sigma_image']
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


# def plot_models(model_geom, ax):
#     ls = dict(disk='-.', bulge=':', bar='--')
#     cs = dict(disk='C0', bulge='C1', bar='C2')
#     try:
#         assert len(ax) >= 3
#     except TypeError:
#         ax = [ax] * 3
#     for i, k in enumerate(('disk', 'bulge', 'bar')):
#         ax[i].add_patch(PolygonPatch(
#             scale(model_geom[k], 3, 3),
#             ec='none', fc=cs[k], alpha=0.4, zorder=3
#         ))
#         ax[i].add_patch(PolygonPatch(
#             scale(model_geom[k], 3, 3),
#             ec='k', fc='none', ls=ls[k], alpha=1, lw=2, zorder=3
#         ))


def cluster_components(models, data, ba, angle):
    model_cluster = ag.cluster_components(
        models=models, image_size=data.shape,
        phi=angle, ba=ba,
    )
    aggregation_result = ag.aggregate_components(model_cluster)
    return model_cluster, aggregation_result


ba = gal['PETRO_BA90']

print('Clustering all components')
# print('- In Zooniverse coordinates')
# t0 = time()
# zoo_result = cluster_components(zoo_models, im, ba, np.rad2deg(gal['montage_angle']))
# print('\tElapsed time: {:.2f}'.format(time() - t0))
print('- In SDSS frame coordinates')
t0 = time()
clusters, agg_model = cluster_components(models, fm['galaxy_data'], ba, np.rad2deg(gal['original_angle']))
print('\tElapsed time: {:.2f}'.format(time() - t0))

# save the results
cluster_output = os.path.join(lib, 'clusters')
os.makedirs(cluster_output, exist_ok=True)
with open(os.path.join(cluster_output, f'{subject_id}.pickle'), 'wb') as f:
    pickle.dump(clusters, f)

raw_agg_model_output = os.path.join(lib, 'agg_models')
os.makedirs(raw_agg_model_output, exist_ok=True)
print(
    'Writing aggregate model to',
    os.path.join(raw_agg_model_output, f'{subject_id}.json')
)
with open(os.path.join(raw_agg_model_output, f'{subject_id}.json'), 'w') as f:
    f.write(pg.make_json(agg_model))


# now to tune the model, this can take some time
print('Tuning model')
agg_model = Model(
    agg_model,
    data,
    psf=psf,
    sigma_image=sigma
)
# res, partially_tuned_model = fit_model(
#     agg_model,
#     params=cfg.SLIDER_FIT_PARAMS,
#     progress=args.progress,
#     options=dict(maxiter=200, disp=(not args.progress))
# )
res, tuned_model = fit_model(
    agg_model,
    params=cfg.FIT_PARAMS,
    progress=args.progress,
    options=dict(maxiter=200, disp=(not args.progress))
)

print(
    'Writing tuned aggregate model to',
    os.path.join(args.output, f'agg/{subject_id}.json')
)

with open(os.path.join(args.output, f'agg/{subject_id}.json'), 'w') as f:
    f.write(pg.make_json(tuned_model))
