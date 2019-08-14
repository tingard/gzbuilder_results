import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from astropy.wcs import WCS
from astropy import log
log.setLevel('ERROR')

parser = argparse.ArgumentParser(
    description=(
        'Generate subsection of the NSA catalogue used by galaxy_utilities'
    )
)
parser.add_argument('--subjects', metavar='/path/to/subjects.csv', required=True,
                    type=str, help='Location of Zooniverse Subject export')

parser.add_argument('--montages', metavar='/path/to/montageOutputs',
                    required=True, type=str,
                    help='Location of Montage Output folder from subject creation')

parser.add_argument('--images', metavar='/path/to/images',
                    required=True, type=str,
                    help='Location of FITS Images downloaded from SkyServer during subject creation')

args = parser.parse_args()

this_file_location = os.path.dirname(os.path.abspath(__file__))

try:
    montages = [f for f in os.listdir(args.montages) if not f[0] == '.']
    montageCoordinates = np.array([
        [float(j) for j in i.replace('+', ' ').split(' ')]
        if '+' in i
        else [float(j) for j in i.replace('-', ' -').split(' ')]
        for i in [f for f in os.listdir(args.montages) if not f[0] == '.']
    ])
except FileNotFoundError:
    print('Could not find montageOutputs, some functions may not work')
    sys.exit(0)

subjects = pd.read_csv(
    args.subjects,
).drop_duplicates(subset='subject_id').set_index('subject_id', drop=False)

metadata = subjects.metadata.apply(json.loads).apply(pd.Series)

try:
    df_nsa = pd.read_pickle(os.path.join(this_file_location, 'df_nsa.pkl'))
except FileNotFoundError:
    print('Could not find df_nsa.pkl, please run "make_nsa_subsection.py"')
    sys.exit(0)


def get_angle(gal, fits_name, image_size=np.array([512, 512])):
    """obtain the galaxy's rotation in Zooniverse image coordinates. This is
    made slightly trickier by some decisions in the subject
    creation pipeline.
    """
    # First, use a WCS object to obtain the rotation in pixel coordinates, as
    # would be obtained from `fitsFile[0].data`
    wFits = WCS(fits_name)
    # edit to center on the galaxy
    wFits.wcs.crval = [float(gal['RA']), float(gal['DEC'])]
    wFits.wcs.crpix = image_size

    r = 4 * float(gal['PETRO_THETA']) / 3600
    phi = float(gal['PETRO_PHI90'])

    center_pix, dec_line = np.array(wFits.all_world2pix(
        [gal['RA'], gal['RA']],
        [gal['DEC'], gal['DEC'] + r],
        0
    )).T

    rot = [
        [np.cos(np.deg2rad(phi)), -np.sin(np.deg2rad(phi))],
        [np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))]
    ]
    vec = np.dot(rot, dec_line - center_pix)
    rotation_angle = 90 - np.rad2deg(np.arctan2(vec[1], vec[0])) - 90
    return rotation_angle


def get_fits_location(gal):
    montagesDistanceMask = np.add.reduce(
        (montageCoordinates - [gal['RA'], gal['DEC']])**2,
        axis=1
    ) < 0.01
    if np.any(montagesDistanceMask):
        # __import__('warnings').warn('Using montaged image')
        montageFolder = montages[
            np.where(montagesDistanceMask)[0][0]
        ]
        fits_name = os.path.join(
            args.montages,
            montageFolder,
            'mosaic.fits',
        )
    else:
        fits_name = os.path.join(
            args.images,
            '{0}/{1}/frame-r-{0:06d}-{1}-{2:04d}.fits'
        ).format(
            int(gal['RUN']),
            int(gal['CAMCOL']),
            int(gal['FIELD'])
        )
    return fits_name


galaxy_info = pd.Series([])
with tqdm(subjects.index.values, desc='Iterating over subjects') as bar:
    for subject_id in bar:
        # Grab the metadata of the subject we are working on
        subject = subjects.loc[subject_id]
        # And the NSA data for the galaxy (if it's a galaxy with NSA data,
        # otherwise throw an error)
        if metadata.loc[subject_id].get('NSA id', False) is not np.nan:
            try:
                gal = df_nsa.drop_duplicates(
                    subset='NSAID'
                ).set_index(
                    'NSAID',
                    drop=False
                ).loc[
                    int(metadata.loc[subject_id]['NSA id'])
                ]
            except KeyError:
                gal = {}
                raise KeyError(
                    'Metadata does not contain valid NSA id (probably an older galaxy)'
                )
            fits_name = get_fits_location(gal)
            angle = get_angle(gal, fits_name, np.array((512, 512))) % 180
            gal['angle'] = angle
            galaxy_info.loc[subject_id] = gal

galaxy_info.apply(pd.Series).to_csv('gal-metadata.csv')
