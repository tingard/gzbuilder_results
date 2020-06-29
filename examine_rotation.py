import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from astropy.wcs import WCS, FITSFixedWarning
from asinh_cmap import asinh_cmap
from lib.make_cutouts import get_frames
import warnings

warnings.simplefilter('ignore', FITSFixedWarning)

fm = pd.read_pickle('lib/fitting_metadata.pkl')
gal_df = pd.read_csv('lib/gal-metadata.csv', index_col=0)


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


def make_vectors(data, angle, radius, q):
    major = np.stack((
        data.shape,
        np.dot(rotation(angle), (0, 2 * radius))
        + data.shape[0]
    )) / 2

    minor = np.stack((
        data.shape,
        np.dot(rotation(angle), (2 * radius * q, 0))
        + data.shape[0]
    )) / 2
    return major, minor


with tqdm(fm.index.values, desc='Iterating over subjects') as bar:
    for subject_id in bar:
        im = np.array(
            Image.open('lib/subject_data/{}/image.png'.format(subject_id))
        )[::-1]

        data = fm['galaxy_data'].loc[subject_id]
        gal = gal_df.loc[subject_id]
        input_frames, montage_frame = get_frames(subject_id=subject_id)
        angles = transform_angle(input_frames[0], montage_frame, gal['PETRO_PHI90'])
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        v, v2 = make_vectors(
            im, angles['montaged'],
            gal['PETRO_THETA'] / 0.396 * im.shape[0] / data.shape[0],
            gal['PETRO_BA90']
        )
        plt.title(angles['montaged'])
        plt.imshow(im, origin='lower', cmap='gray')
        plt.plot(*v.T, 'r')
        plt.plot(*v2.T, 'g')

        plt.subplot(122)
        plt.title(angles['original'])
        v, v2 = make_vectors(
            data, angles['original'],
            gal['PETRO_THETA'] / 0.396, gal['PETRO_BA90']
        )
        plt.imshow(data, origin='lower', cmap=asinh_cmap)
        plt.plot(*v.T, 'r')
        plt.plot(*v2.T, 'g')

        plt.savefig('rotation_test/{}.png'.format(subject_id), bbox_inches='tight')
        plt.close()
