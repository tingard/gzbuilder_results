import os
import numpy as np
try:
    import modin.pandas as pd
except ImportError:
    import pandas as pd
import galaxy_utilities as gu
from tqdm import tqdm
import make_cutouts as mkct
from astropy.wcs import FITSFixedWarning
import warnings

warnings.simplefilter('ignore', FITSFixedWarning)
# we'll be dividing by NaN a lot (regions of zero pixel coverage)
warnings.simplefilter('ignore', RuntimeWarning)

loc = os.path.abspath(os.path.dirname(__file__))
sid_list = np.loadtxt(os.path.join(loc, 'subject-id-list.csv'), dtype='u8')


def get_data(subject_id):
    diff_data = gu.get_diff_data(subject_id)

    original_image = np.array(diff_data['imageData'])
    pixel_mask = 1 - np.array(diff_data['mask'])

    # generate a cutout and sigma image
    I, sigma_I = mkct.get_image_and_error(subject_id)

    # scale the image and std to match that used in modelling
    im_scaled = I / diff_data['multiplier']
    sd_scaled = sigma_I / diff_data['multiplier']

    # make sure arrays are all of the correct size
    assert np.all(np.equal(im_scaled.shape, sd_scaled.shape))

    # mask out regions in the cutout outside the frame
    try:
        pixel_mask[np.isnan(im_scaled)] = 0
        pixel_mask[np.isnan(sd_scaled)] = 0
        # # could just leave as NaNs?
        # im_scaled[np.isnan(im_scaled)] = 0
        # sd_scaled[np.isnan(im_scaled)] = np.inf
    except ValueError as e:
        print(im_scaled.shape, pixel_mask.shape)
        raise e

    pixel_mask = pixel_mask.astype(int)
    np.nanprod((im_scaled, pixel_mask), axis=0)
    if np.any(np.isnan(np.nanprod((im_scaled, pixel_mask), axis=0))):
        raise ValueError('NaNs still present in image outside mask')
    if np.any(np.isnan(np.nanprod((sd_scaled, pixel_mask), axis=0))):
        raise ValueError('NaNs still present in sigma image outside mask')

    return dict(
        psf=gu.get_psf(subject_id),
        pixel_mask=pixel_mask * pixel_mask,
        galaxy_data=im_scaled,
        original_data=original_image,
        sigma_image=sd_scaled,
        width=diff_data['width'],
        size_diff=diff_data['width'] / diff_data['imageWidth'],
    )


tqdm.pandas(desc='Making DataFrame')

df = pd.Series(sid_list, index=sid_list)
df = df.progress_apply(get_data)
# with tqdm(sid_list, desc='Making DataFrame') as bar:
#     for subject_id in bar:
#         df[subject_id] = get_data(subject_id)

df = df.apply(pd.Series)
df.to_pickle(os.path.join(loc, 'fitting_metadata.pkl'))
