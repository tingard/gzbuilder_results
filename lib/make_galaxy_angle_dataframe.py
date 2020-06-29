import os
import sys
import re
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy import log
log.setLevel('ERROR')


this_file_location = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(
    description=(
        'Generate subsection of the NSA catalogue used by galaxy_utilities. '
        'Requires make_nsa_subsection.py to have been run, and the output '
        'df-nsa.pkl to be present in the same folder as this file. '
        'Also requires a Zooniverse Subject export, and the locations of the '
        'FITS files used in the subject creation process (fits headers are '
        'required for rotations)'
    )
)
default_subjects_csv_location = os.path.join(
    this_file_location,
    'galaxy-builder-subjects.csv'
)
default_montages_location = os.path.join(
    this_file_location,
    '../../gzbuilder_data_prep/montageOutputs'
)

default_subjects_location = os.path.join(
    this_file_location,
    '../../gzbuilder_data_prep/fits_images'
)

parser.add_argument('--subjects', metavar='/path/to/subjects.csv',
                    default=default_subjects_csv_location,
                    type=str, help='Location of Zooniverse Subject export')

parser.add_argument('--montages', metavar='/path/to/montageOutputs',
                    default=default_montages_location, type=str,
                    help='Location of Montage Output folder from subject creation')

parser.add_argument('--images', metavar='/path/to/images',
                    default=default_subjects_location, type=str,
                    help='Location of FITS Images downloaded from SkyServer during subject creation')

args = parser.parse_args()

montage_groups = [
    i for i in os.listdir(args.montages)
    if re.match(r'([0-9.]+)\+([\-0-9.]+)', i) is not None
]

# create a list of coordinates to search
montage_coords = SkyCoord(
    np.array([
        i.groups() for i in (
            re.match(r'([0-9.]+)\+([\-0-9.]+)', i)
            for i in montage_groups
        )
        if i is not None
    ]).astype(float),
    unit=u.degree,
)

subjects = pd.read_csv(
    args.subjects,
).drop_duplicates(subset='subject_id').set_index('subject_id', drop=False)

metadata = subjects.metadata.apply(json.loads).apply(pd.Series)

try:
    df_nsa = pd.read_pickle(os.path.join(this_file_location, 'df_nsa.pkl'))
except FileNotFoundError:
    print('Could not find df_nsa.pkl, please run "make_nsa_subsection.py"')
    sys.exit(0)


def rotation(a):
    return np.array(((np.cos(a), np.sin(a)), (-np.sin(a), np.cos(a))))


def transform_angle(original_frame, montage_frame, angle):
    wcs = dict(original=WCS(original_frame), montaged=WCS(montage_frame))
    angles = dict()
    for k in wcs.keys():
        centre = wcs[k].wcs.crval.copy()
        dec_line = centre + (10/3600, 0)
        # negative as NSA uses (N to E) angles
        line_of_angle = np.dot(
            rotation(np.deg2rad(angle)),
            dec_line - centre
        ) + centre
        centre_pix, angle_line_pix = wcs[k].all_world2pix(
            [centre, line_of_angle],
            0
        )
        new_angle = np.arctan2(*np.flip(angle_line_pix - centre_pix))
        angles[k] = -new_angle
    return angles


def get_input_frames(gal):
    gal_coord = SkyCoord(gal['RA'], gal['DEC'], unit=u.degree)
    idx = np.argmin(montage_coords.separation(gal_coord))
    fpath = os.path.join(args.montages, montage_groups[idx])
    return [
        os.path.join(fpath, i)
        for i in os.listdir(fpath)
        if '.fits' in i
    ]


def get_output_frame(input_frames):
    if len(input_frames) == 1:
        return input_frames[0]
    os.path.join(os.path.dirname(input_frames[0]), 'mosaic.fits')
    return os.path.join(
        os.path.dirname(input_frames[0]),
        'mosaic.fits'
    ).replace('montageGroups', 'montageOutputs')


def get_frames(gal):
    input_frames = get_input_frames(gal)
    output_frame = get_output_frame(input_frames)
    return input_frames, output_frame


if __name__ == '__main__':
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
                original_fits_names, montage_fits_name = get_frames(gal)
                try:
                    angles = transform_angle(
                        original_fits_names[0],
                        montage_fits_name,
                        gal['PETRO_PHI90']
                    )
                except OSError as e:
                    print(subject_id, e)
                    continue
                gal['montage_angle'] = angles['montaged']
                gal['original_angle'] = angles['original']
                # legacy
                gal['angle'] = angles['montaged']
                galaxy_info.loc[subject_id] = gal

    galaxy_info.apply(pd.Series).to_csv('gal-metadata.csv')
