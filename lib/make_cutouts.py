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
from astropy.wcs import WCS
from astropy.nddata.utils import NoOverlapError
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from tqdm import tqdm
import sep
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
    os.path.join(os.path.dirname(input_frames[0]), 'mosaic.fits')
    return os.path.join(
        os.path.dirname(input_frames[0]),
        'mosaic.fits'
    ).replace('montageGroups', 'montageOutputs')


def get_frames(gal=None, subject_id=None):
    if gal is None and subject_id is None:
        raise TypeError('Mising required argument of gal object or subject_id')
    if subject_id is not None:
        gal = gu.get_galaxy(subject_id)
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


def get_montaged_cutout(subject_id):
    gal = gu.get_galaxy(subject_id)
    ra, dec = gu.metadata.loc[subject_id][['ra', 'dec']]
    centre_pos, dx, dy = get_cutout_params(gal, ra, dec)
    montage_fits = fits.open(get_frames(subject_id=subject_id)[1])
    montage_cutout = Cutout2D(
        montage_fits[0].data,
        centre_pos,
        (dx, dy),
        wcs=WCS(montage_fits[0]),
        mode='partial',
        copy=True,
    )
    return montage_cutout


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
    darkvar = darkvar_r.loc[img.header['camcol']]
    nelec = dn * gain
    return {'nelec': nelec, 'calib': calib_img, 'sky': sky_img,
            'gain': gain, 'darkvar': darkvar}


def get_frame_data(subject_id, verbose=False, ensure_consistent_sizes=True):
    gal, _ = gu.get_galaxy_and_angle(subject_id)
    input_frames_path, _ = get_frames(gal)

    # use the positions from the subject metadata as it matches the ones shown
    # to volunteers
    centre = gu.metadata.loc[subject_id][['ra', 'dec']].astype(float)
    centre_pos, dx, dy = get_cutout_params(gal, *centre)
    cutout_size = (dx, dy)

    # do the first cutout and reproject
    frames = pd.DataFrame(
        [],
        columns=('frame', 'image', 'nelec', 'calib', 'sigma', 'sky', 'wcs',
                 'gain', 'darkvar')
    )

    with tqdm(input_frames_path, leave=False) as bar:
        for i, frame in enumerate(bar):
            f = fits.open(frame)
            frame_wcs = WCS(f[0])
            data = get_data(f)
            gain = data['gain']
            darkvar = data['darkvar']

            def make_cutout(data):
                return Cutout2D(
                    data,
                    centre_pos,
                    cutout_size,
                    wcs=frame_wcs,
                    mode='partial',
                    copy=True,
                )
            try:
                img_cutout = make_cutout(f[0].data)
                nelec_cutout = make_cutout(data['nelec'])
                calib_cutout = make_cutout(data['calib'])
                sky_cutout = make_cutout(data['sky'])
            except NoOverlapError as e:
                if verbose:
                    print(e)
                    print(frame)
                continue

            if ensure_consistent_sizes:
                cutout_size = img_cutout.data.shape
            coverage_mask = np.isfinite(nelec_cutout.data)
            if np.any(coverage_mask):
                sigma = (
                    calib_cutout.data
                    * np.sqrt(nelec_cutout.data / gain**2 + darkvar)
                )
                coverage_mask[~coverage_mask] = np.nan
                frames.loc[i] = {
                    'frame': frame,
                    'image': img_cutout.data,
                    'nelec': nelec_cutout.data,
                    'calib': calib_cutout.data,
                    'sigma': sigma,
                    'sky': sky_cutout.data,
                    'wcs': img_cutout.wcs,
                    'gain': coverage_mask.astype(int) * gain,
                    'darkvar': coverage_mask.astype(int) * darkvar,
                }
    return frames


def stack_frames(frames):
    elec_stack = np.stack(frames['nelec'].values)
    calib_stack = np.stack(frames['calib'].values)
    sky_stack = np.stack(frames['sky'].values)
    g_stack = np.stack(frames['gain'].values)
    v_stack = np.stack(frames['darkvar'].values)
    pixel_count = np.isfinite(elec_stack).astype(int).sum(axis=0).astype(float)

    # we can just stack the images
    I_mean = np.nanmean(np.stack(frames['image'].values), axis=0)

    # or calculate from the raw data
    # I = C(n / g - S)
    I_combined = np.nansum(
        calib_stack * ((elec_stack / g_stack) - sky_stack),
        axis=0
    ) / pixel_count

    # if this fails then something is wrong with the data
    if not np.allclose(I_mean, I_combined):
        raise ValueError('Inconsistencies in data')

    # sigma = 1/N * sqrt(sum(C**2 (n / g**2) + v))
    sigma_I = np.sqrt(
        np.nansum(
            calib_stack**2 * ((elec_stack / g_stack**2) + v_stack),
            axis=0
        )
    ) / pixel_count
    return I_combined, sigma_I


def sourceExtractImage(data, bkgArr=None, sortType='center', verbose=False,
                       **kwargs):
    """Extract sources from data array and return enumerated objects sorted
    smallest to largest, and the segmentation map provided by source extractor
    """
    data = np.array(data).byteswap().newbyteorder()
    if bkgArr is None:
        bkgArr = np.zeros(data.shape)
    o = sep.extract(data, kwargs.pop('threshold', 0.05), segmentation_map=True,
                    **kwargs)
    if sortType == 'size':
        if verbose:
            print('Sorting extracted objects by radius from size')
        sizeSortedObjects = sorted(
            enumerate(o[0]), key=lambda src: src[1]['npix']
        )
        return sizeSortedObjects, o[1]
    elif sortType == 'center':
        if verbose:
            print('Sorting extracted objects by radius from center')
        centerSortedObjects = sorted(
            enumerate(o[0]),
            key=lambda src: (
                (src[1]['x'] - data.shape[0] / 2)**2
                + (src[1]['y'] - data.shape[1] / 2)**2
            )
        )[::-1]
        return centerSortedObjects, o[1]


def maskArr(arrIn, segMap, maskID):
    """Return a true/false mask given a segmentation map and segmentation ID
    True signifies the pixel should be masked
    """
    return np.logical_and(segMap != maskID, segMap != 0)


def generate_new_cutout(subject_id, frame_data=None):
    if frame_data is None:
        frame_data = get_frame_data(subject_id)
    stacked_image, sigma_image = stack_frames(frame_data)
    # prevents astropy.io.fits reads things in a big-endian way, which makes
    # MacOS sad
    image_data = stacked_image.byteswap().newbyteorder()
    # source extract the image for masking
    objects, segmentation_map = sourceExtractImage(image_data)
    mask = maskArr(image_data, segmentation_map, objects[-1][0] + 1)
    mask[np.isnan(image_data)] = True
    masked_image = np.ma.masked_array(stacked_image, mask)
    masked_sigma = np.ma.masked_array(sigma_image, mask)
    return masked_image, masked_sigma
