import os
import numpy as np
import pandas as pd
import galaxy_utilities as gu
from tqdm import tqdm
loc = os.path.abspath(os.path.dirname(__file__))
sid_list = np.loadtxt(os.path.join(loc, 'subject-id-list.csv'), dtype='u8')
best_models = pd.read_pickle(os.path.join(loc, 'best_individual.pickle'))

df = pd.Series([])
with tqdm(sid_list, desc='Making DataFrame', leave=False) as bar:
    for subject_id in bar:
      diff_data = gu.get_diff_data(subject_id)
      df[subject_id] = dict(
          psf=gu.get_psf(subject_id),
          pixel_mask=1 - np.array(diff_data['mask']),
          galaxy_data=np.array(diff_data['imageData']),
          image_size=diff_data['width'],
          m=best_models['Model'].loc[subject_id],
      )

df = df.apply(pd.Series)
df.to_pickle(os.path.join(loc, 'fitting_metadata.pkl'))
