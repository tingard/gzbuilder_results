import pandas as pd
import lib.galaxy_utilities as gu

df = gu.metadata.reset_index().groupby('SDSS dr7 id')['subject_id'].unique()
df = df.apply(pd.Series)
df.dropna().to_csv('lib/duplicate_galaxies.csv')
