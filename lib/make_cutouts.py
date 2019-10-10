# - Our frames are not aligned.
# - Calculate $n_i$, the electron counts for each frame
# - For each frame, create a slightly larger than required cutout of nelec
#   and the Calibration image
# - Use `reproject` to align the electron counts and the calibration images of
#   each frame to the WCS of the FITS header of the `Montage`-created image
#   (which is what volunteer models were drawn on).
# - Proceed with the error calculation
#   - Note that $N$ will not be the same for each pixel, as some regions of the
#     image may be covered by different numbers of frames
# - Once we have $\bar{I}$ and $\sigma_I$, perform a cutout of the required size
#   for each and return!
#
# n.b. MAKE LOTS OF SAVE POINTS (i.e. write out the cutout + reprojected FITS
# files)


import os
import re
import numpy as np
from scipy.interpolate import interp2d
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata.utils import NoOverlapError
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from collections import namedtuple
from tqdm import tqdm
import reproject

try:
    import galaxy_utilities as gu
except ModuleNotFoundError:
    import lib.galaxy_utilities as gu

gain_table = pd.DataFrame(
    [
        [1.62, 3.32, 4.71, 5.165, 4.745],
        [np.nan, 3.855, 4.6, 6.565, 5.155],
        [1.59, 3.845, 4.72, 4.86, 4.885],
        [1.6, 3.995, 4.76, 4.885, 4.775],
        [1.47, 4.05, 4.725, 4.64, 3.48],
        [2.17, 4.035, 4.895, 4.76, 4.69],
    ],
    index=pd.Series(np.arange(6) + 1, name='camcol'),
    columns=list('ugriz')
)
darkvar_r = pd.Series(
    [1.8225, 1.00, 1.3225, 1.3225, 0.81, 0.9025],
    index=np.arange(6) + 1,
    name='darkvar_r'
)

GetCutoutsOutput = namedtuple(
    'mosiac_result',
    ['individual_cutouts', 'image', 'sigma_image', 'pixel_counts']
)

# filter directory for bad files (ie .DS_Store)
loc = os.path.abspath(os.path.dirname(__file__))
montage_output_path = os.path.join(
    loc,
    '../../gzbuilder_data_prep/montageGroups/'
)
montage_groups = [
    i for i in os.listdir(montage_output_path)
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


def get_input_frames(gal):
    gal_coord = SkyCoord(gal['RA'], gal['DEC'], unit=u.degree)
    idx = np.argmin(montage_coords.separation(gal_coord))
    fpath = os.path.join(montage_output_path, montage_groups[idx])
    return [
        os.path.join(fpath, i)
        for i in os.listdir(fpath)
        if '.fits' in i
    ]


def get_output_frame(input_frames):
    if len(input_frames) == 1:
        return input_frames[0]
    return os.path.join(
        '/', *input_frames[0].split('/')[:-1], 'mosaic.fits'
    ).replace('montageGroups', 'montageOutputs')


def get_frames(gal):
    input_frames = get_input_frames(gal)
    output_frame = get_output_frame(input_frames)
    return input_frames, output_frame


def get_cutout_params(gal, ra, dec):
    centre_pos = SkyCoord(
        ra, dec,
        unit=u.degree, frame='fk5'
    )
    dx = 4 * float(gal['PETRO_THETA']) * u.arcsec
    dy = 4 * float(gal['PETRO_THETA']) * u.arcsec
    return centre_pos, dx, dy


def get_data(f):
    img = f[0]
    ff = f[1]
    sky = f[2]
    allsky, xinterp, yinterp = sky.data[0]
    sky_img = interp2d(
        np.arange(allsky.shape[1]),
        np.arange(allsky.shape[0]),
        allsky,
    )(xinterp, yinterp)
    calib_img = np.tile(np.expand_dims(ff.data, 1), img.data.shape[0]).T
    dn = img.data / calib_img + sky_img
    gain = gain_table.loc[img.header['camcol']][img.header['FILTER']]
    # darkvar = darkvar_r.loc[img.header['camcol']]
    nelec = dn * gain
    return {'nelec': nelec, 'calib': calib_img, 'sky': sky_img}


def get_image_and_error(subject_id, verbose=False):
    gal, _ = gu.get_galaxy_and_angle(subject_id)
    input_frames_path, output_mosiac_path = get_frames(gal)
    output_mosaic = fits.open(output_mosiac_path)

    # use the positions from the subject metadata as it matches the ones shown
    # to volunteers
    ra, dec = gu.metadata.loc[subject_id][['ra', 'dec']]

    centre_pos, dx, dy = get_cutout_params(gal, ra, dec)
    cutout_size = (dx, dy)
    large_cutout_size = (dx * 1.25, dy * 1.25)

    target_cutout = Cutout2D(
        output_mosaic[0].data,
        centre_pos,
        large_cutout_size,
        wcs=WCS(output_mosaic[0].header),
        mode='partial',
        copy=True,
    )

    # do the first cutout and reproject
    reprojection_results = pd.DataFrame(
        [],
        columns=('nelec', 'calib', 'sky', 'wcs', 'gain', 'darkvar')
    )
    with tqdm(input_frames_path, leave=False) as bar:
        for i, frame in enumerate(bar):
            f = fits.open(frame)
            frame_wcs = WCS(f[0])
            gain = gain_table.loc[f[0].header['camcol']][f[0].header['FILTER']]
            darkvar = darkvar_r.loc[f[0].header['camcol']]
            data = get_data(f)

            # make large cutouts of the needed images
            def make_large_cutout(arr):
                return Cutout2D(
                    arr,
                    centre_pos,
                    large_cutout_size,
                    wcs=frame_wcs,
                    mode='partial',
                    copy=True,
                )

            try:
                large_nelec_cutout = make_large_cutout(data['nelec'])
                large_calib_cutout = make_large_cutout(data['calib'])
                large_sky_cutout = make_large_cutout(data['sky'])
            except NoOverlapError as e:
                if verbose:
                    print(e)
                    print(frame)
                continue
            # reproject the cutouts to the target wcs
            reproj_nelec, coverage_nelec = reproject.reproject_exact(
                (large_nelec_cutout.data, large_nelec_cutout.wcs),
                target_cutout.wcs,
                shape_out=target_cutout.data.shape
            )
            reproj_calib, coverage_calib = reproject.reproject_exact(
                (large_calib_cutout.data, large_calib_cutout.wcs),
                target_cutout.wcs,
                shape_out=target_cutout.data.shape
            )
            reproj_sky, coverage_sky = reproject.reproject_exact(
                (large_sky_cutout.data, large_sky_cutout.wcs),
                target_cutout.wcs,
                shape_out=target_cutout.data.shape
            )

            def make_cutout(arr):
                return Cutout2D(
                    arr,
                    centre_pos,
                    cutout_size,
                    wcs=target_cutout.wcs,
                    mode='partial',
                    copy=True,
                )
            nelec_cutout = make_cutout(reproj_nelec)
            calib_cutout = make_cutout(reproj_calib)
            sky_cutout = make_cutout(reproj_sky)

            coverage_mask = np.isfinite(nelec_cutout.data)
            coverage_mask[~coverage_mask] = np.nan
            if np.any(coverage_mask):
                reprojection_results.loc[i] = {
                    'nelec': nelec_cutout.data,
                    'calib': calib_cutout.data,
                    'sky': sky_cutout.data,
                    'wcs': nelec_cutout.wcs,
                    'gain': coverage_mask.astype(int) * gain,
                    'darkvar': coverage_mask.astype(int) * darkvar
                }

    # we now have a DataFrame of electron counts and calibration images
    elec_stack = np.stack(reprojection_results['nelec'].values)
    calib_stack = np.stack(reprojection_results['calib'].values)
    sky_stack = np.stack(reprojection_results['sky'].values)
    g_stack = np.stack(reprojection_results['gain'].values)
    v_stack = np.stack(reprojection_results['darkvar'].values)
    pixel_count = np.isfinite(elec_stack).astype(int).sum(axis=0).astype(float)
    # I = C(n / g - S)
    I_combined = np.nansum(
        calib_stack * (elec_stack / g_stack - sky_stack),
        axis=0
    ) / pixel_count
    sigma_I = np.sqrt(
        np.nansum(calib_stack**2 * ((elec_stack / g_stack**2) + v_stack), axis=0)
    ) / pixel_count
    return I_combined, sigma_I


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import warnings
    warnings.simplefilter('ignore', FITSFixedWarning)
    warnings.simplefilter('ignore', FutureWarning)

    subject_id = 20902040
    I, sigma_I = get_image_and_error(subject_id)

    diff_data = gu.get_diff_data(subject_id)
    pixel_mask = 1 - np.array(diff_data['mask'])
    pixel_mask[pixel_mask == 0] = np.nan

    original_image = np.array(diff_data['imageData']) * pixel_mask

    I_scaled = I / diff_data['multiplier'] * pixel_mask

    diff = I_scaled - original_image

    print('Old:', np.nanmin(original_image), np.nanmax(original_image))
    print('New:', np.nanmin(I_scaled), np.nanmax(I_scaled))

    lims = dict(
        vmin=-max(np.nanmax(np.abs(I_scaled)), np.nanmax(np.abs(original_image))),
        vmax=max(np.nanmax(np.abs(I_scaled)), np.nanmax(np.abs(original_image))),
        cmap='coolwarm'
    )
    plt.imshow(I_scaled)
    plt.colorbar()
    plt.figure()
    plt.imshow(sigma_I / diff_data['multiplier'] * pixel_mask)
    plt.colorbar()
    plt.show()
