"""Arguably the most important file in the data analysis process. This script
is used to obtain fitting data from galaxy builder metadata and subject files.
"""

import os
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import json
import requests
from PIL import Image
from astropy.wcs import WCS
from gzbuilder_analysis.spirals import get_drawn_arms as __get_drawn_arms
from gzbuilder_analysis.spirals import deprojecting as dpj
from shapely.geometry import box, Point
from shapely.affinity import rotate as shapely_rotate, scale as shapely_scale


# needed when we want to load data files relative to this file's location, not
# the current working directory
def get_path(s):
    return os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        s
    )
try:
    df_nsa = pd.read_pickle(get_path('df_nsa.pkl'))
except FileNotFoundError:
    print('Could not locate df_nsa.pkl, please run make_nsa_subsection.py')
try:
    gal_angle_df = pd.read_csv(get_path('gal-metadata.csv'), index_col=0)
except FileNotFoundError:
    print('Could not locate gal-metadata.csv, please run make_galaxy_angle_dataframe.py')

classifications = pd.read_csv(
    get_path('galaxy-builder-classifications.csv')
)

subjects = pd.read_csv(
    get_path('galaxy-builder-subjects.csv')
).drop_duplicates(subset='subject_id').set_index('subject_id', drop=False)
metadata = subjects.metadata.apply(json.loads)

# # Some galaxies were montaged when created. Create a list of their coordinates
# # for use later
# try:
#     montage_output_path = get_path('montageOutputs')
#     montages = [f for f in os.listdir(montage_output_path) if not f[0] == '.']
#     montageCoordinates = np.array([
#         [float(j) for j in i.replace('+', ' ').split(' ')]
#         if '+' in i
#         else [float(j) for j in i.replace('-', ' -').split(' ')]
#         for i in [f for f in os.listdir(montage_output_path) if not f[0] == '.']
#     ])
# except FileNotFoundError:
#     print('Could not find montageOutputs, some functions may not work')
#     montage_output_path = None
#
#
# def get_fits_location(gal):
#     montagesDistanceMask = np.add.reduce(
#         (montageCoordinates - [gal['RA'], gal['DEC']])**2,
#         axis=1
#     ) < 0.01
#     if np.any(montagesDistanceMask):
#         # __import__('warnings').warn('Using montaged image')
#         montageFolder = montages[
#             np.where(montagesDistanceMask)[0][0]
#         ]
#         fits_name = get_path('{}/{}/{}'.format(
#             'montageOutputs',
#             montageFolder,
#             'mosaic.fits'
#         ))
#     else:
#         fileTemplate = get_path(
#             'fitsImages/{0}/{1}/frame-r-{0:06d}-{1}-{2:04d}.fits'
#         )
#         fits_name = fileTemplate.format(
#             int(gal['RUN']),
#             int(gal['CAMCOL']),
#             int(gal['FIELD'])
#         )
#     return fits_name
#
#
# def get_angle(gal, fits_name, image_size=np.array([512, 512])):
#     wFits = WCS(fits_name)
#     # edit to center on the galaxy
#     wFits.wcs.crval = [float(gal['RA']), float(gal['DEC'])]
#     wFits.wcs.crpix = image_size
#
#     r = 4 * float(gal['PETRO_THETA']) / 3600
#     phi = float(gal['PETRO_PHI90'])
#
#     center_pix, dec_line = np.array(wFits.all_world2pix(
#         [gal['RA'], gal['RA']],
#         [gal['DEC'], gal['DEC'] + r],
#         0
#     )).T
#
#     rot = [
#         [np.cos(np.deg2rad(phi)), -np.sin(np.deg2rad(phi))],
#         [np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))]
#     ]
#     vec = np.dot(rot, dec_line - center_pix)
#     rotation_angle = 90 - np.rad2deg(np.arctan2(vec[1], vec[0])) - 90
#     return rotation_angle


def get_galaxy_and_angle(subject_id, imShape=(512, 512)):
    gal = gal_angle_df.loc[subject_id]
    return gal, gal['angle']


def get_drawn_arms(subject_id, classifications=classifications):
    try:
        qs = ' or '.join('subject_ids == {}'.format(i) for i in subject_id)
    except TypeError:
        qs = 'subject_ids == {}'.format(subject_id)
    return __get_drawn_arms(
        classifications.query(qs)
    )


# def get_ds9_region(gal, fits_name):
#     s = """# Region file format: DS9 version 4.1
#     global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" \
# select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
#     fk5
#     ellipse({},{},{}",{}",{})"""
#     with open(get_path('regions/{}.reg'.format(id)), 'w') as f:
#         f.write(s.format(
#             float(gal['RA']),
#             float(gal['DEC']),
#             float(gal['PETRO_THETA'] * gal['SERSIC_BA']),
#             float(gal['PETRO_THETA']),
#             float(gal['PETRO_PHI90'])
#         ))
#     print(
#         ' '.join((
#             'ds9 {0}',
#             '-regions {1}',
#             '-pan to {2} {3} wcs degrees',
#             '-crop {2} {3} {4} {4} wcs degrees',
#             '-asinh -scale mode 99.5'
#         )).format(
#             fits_name,
#             get_path('regions/{}.reg'.format(id)),
#             gal['RA'],
#             gal['DEC'],
#             gal['PETRO_THETA'] * 2 / 3600,
#         ),
#     )
#
#
# def get_image_data(subject_id):
#     return get_diff_data(subject_id)


def get_image(subject_id):
    image_path = 'subject_data/{}/image.png'.format(subject_id)
    return Image.open(get_path(image_path))


def get_deprojected_image(subject_id, ba, angle):
    return dpj.deproject_array(
        np.array(get_image(subject_id)),
        angle, ba,
    )


def get_diff_data(subject_id):
    diff_path = 'subject_data/{}/diff.json'.format(subject_id)
    with open(get_path(diff_path)) as f:
        diff = json.load(f)
    return {
        **diff,
        **{k: np.array(diff[k], 'f8') for k in ('psf', 'imageData')},
    }


def get_psf(subject_id):
    model_path = 'subject_data/{}/model.json'.format(subject_id)
    with open(get_path(model_path)) as f:
        model = json.load(f)
    return np.array(model['psf'], 'f8')


def bar_geom_from_zoo(a):
    b = box(
        a['x'],
        a['y'],
        a['x'] + a['width'],
        a['y'] + a['height']
    )
    return shapely_rotate(b, a['angle'])


def ellipse_geom_from_zoo(a):
    ellipse = shapely_rotate(
        shapely_scale(
            Point(a['x'], a['y']).buffer(1.0),
            xfact=a['rx'],
            yfact=a['ry']
        ),
        -a['angle']
    )
    return ellipse
