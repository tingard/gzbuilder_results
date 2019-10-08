import os
import re
import numpy as np
from scipy.interpolate import interp2d
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from collections import namedtuple
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
montage_output_path = os.path.join(loc, '../../gzbuilder_data_prep/montageGroups/')
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


def get_montage_frames(gal):
    gal_coord = SkyCoord(gal['RA'], gal['DEC'], unit=u.degree)
    idx = np.argmin(montage_coords.separation(gal_coord))
    fpath = os.path.join(montage_output_path, montage_groups[idx])
    return [
        os.path.join(fpath, i)
        for i in os.listdir(fpath)
        if '.fits' in i
    ]


def get_sigma_image(f):
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
    dn_err = np.sqrt(dn / gain + darkvar)
    img_err = dn_err * calib_img
    return img_err


def get_cutout_params(gal, ra, dec):
    centre_pos = SkyCoord(
        ra, dec,
        unit=u.degree, frame='fk5'
    )
    dx = 4 * float(gal['PETRO_THETA']) * u.arcsec
    dy = 4 * float(gal['PETRO_THETA']) * u.arcsec
    return centre_pos, dx, dy


def get_reprojected_data(frame_loc, output_mosiac, gal, ra, dec):
    # open the file
    frame = fits.open(frame_loc)
    # extract the HDU
    hdu = frame[0]
    # calculate a WCS object
    wcs = WCS(hdu.header)
    # Define cutout positions
    centre_pos, dx, dy = get_cutout_params(gal, ra, dec)
    # # Grab the cutout
    # cutout_im = Cutout2D(
    #     hdu.data,
    #     centre_pos,
    #     (dx, dy),
    #     wcs=wcs,
    #     mode='partial',
    #     copy=True,
    # )
    # # Update the HDU with the new data and WCS
    # hdu.data = cutout_im.data
    # hdu.header.update(cutout_im.wcs.to_header())
    # Reproject the HDU to the montaged coordinate space
    array, _ = reproject.reproject_interp(
        hdu,
        output_mosiac[0].header
    )
    im = Cutout2D(
        array,
        centre_pos,
        (dx, dy),
        wcs=WCS(output_mosiac[0].header),
        mode='partial',
        copy=True,
    )
    # reload the frame
    frame = fits.open(frame_loc)
    sigma = get_sigma_image(frame)
    # cutout_sigma = Cutout2D(
    #     sigma,
    #     centre_pos,
    #     (dx, dy),
    #     wcs=wcs,
    #     mode='partial',
    #     copy=True,
    # )
    hdu.data = sigma  # cutout_sigma.data
    # hdu.header.update(cutout_sigma.wcs.to_header())
    # Reproject the HDU to the montaged coordinate space
    sigma_array, _ = reproject.reproject_interp(
        hdu,
        output_mosiac[0].header
    )
    sigma_cutout = Cutout2D(
        sigma_array,
        centre_pos,
        (dx, dy),
        wcs=WCS(output_mosiac[0].header),
        mode='partial',
        copy=True,
    )
    return im.data, sigma_cutout.data


def get_cutouts(subject_id, verbose=False):
    gal = gu.get_galaxy_and_angle(subject_id)[0]
    input_frames_path = get_montage_frames(gal)
    output_mosiac_path = os.path.join(
        '/', *get_montage_frames(gal)[0].split('/')[:-1], 'mosaic.fits'
    ).replace('montageGroups', 'montageOutputs')
    try:
        output_mosiac = fits.open(output_mosiac_path)
    except FileNotFoundError:
        assert len(input_frames_path) == 1
        output_mosiac = fits.open(input_frames_path[0])

    ra, dec = gu.metadata.loc[subject_id][['ra', 'dec']]

    output = pd.Series([])
    # p = Pool(2)
    # p.starmap()
    for i, frame in enumerate(input_frames_path):
        if verbose:
            print('Working on', frame.split('/')[-1])
        try:
            im, sigma = get_reprojected_data(
                frame,
                output_mosiac,
                gal,
                ra,
                dec
            )
            output.loc[i] = {'image': im, 'sigma_image': sigma}
        except ValueError as e:
            if verbose:
                print(e)
                print('Frame:', frame)

    output = output.apply(pd.Series)

    im_stack = np.stack(output['image'].apply(lambda a: a.data).values)
    im_mean = np.nanmean(im_stack, axis=0)
    pixel_counts = np.isfinite(im_stack).astype(int).sum(axis=0).astype(float)
    # pixel_counts[pixel_counts == 0] == np.nan

    sigma_summed = np.sqrt(
        np.nansum(
            np.stack(
                output['sigma_image'].apply(lambda a: a.data).values
            )**2,
            axis=0,
        )
    )
    sigma_mean = sigma_summed / pixel_counts
    return GetCutoutsOutput(output, im_mean, sigma_mean, pixel_counts)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import warnings
    from astropy.wcs import FITSFixedWarning

    warnings.simplefilter('ignore', FITSFixedWarning)

    subject_id = 21097001

    im_zoo = gu.get_image(subject_id)
    diff_data = gu.get_diff_data(subject_id)
    mask = (1 - np.array(diff_data['mask'])).astype(bool)
    data_zoo = np.array(diff_data['imageData']) * diff_data['multiplier']
    output = get_cutouts(subject_id, verbose=True)

    lims = np.min(data_zoo), np.max(data_zoo)

    print('Making plots')
    f, ax = plt.subplots(ncols=2, nrows=3, figsize=(9, 12), dpi=100)
    ax[0][0].set_title('Zooniverse Image')
    ax[0][0].imshow(im_zoo, cmap='gray')
    ax[0][1].set_title('Montaged Image (Old method)')
    ax[0][1].imshow(data_zoo, vmin=lims[0], vmax=lims[1])
    ax[1][0].set_title('Montaged Image (New method)')
    ax[1][0].imshow(output.image * mask, vmin=lims[0], vmax=lims[1])
    ax[1][1].set_title('Sigma Image')
    ax[1][1].imshow(output.sigma_image * mask)
    ax[2][0].set_title('Pixel mask')
    ax[2][0].imshow(mask)
    ax[2][1].set_title('Pixel counts')
    ax[2][1].imshow(output.pixel_counts, vmin=0, vmax=8, cmap='Set1')

    plt.tight_layout()
    plt.savefig('custom_montage_results.png', bbox_inches='tight')
