import re
import numpy as np
import pandas as pd
import argparse
from astropy.coordinates import SkyCoord
import astropy.units as u
import galaxy_utilities as gu


parser = argparse.ArgumentParser(
    description=(
        'Fit Aggregate model and best individual'
        ' model for a galaxy builder subject'
    )
)
parser.add_argument('--nb1', metavar='/path/to/file.dat', default='',
                    type=str, help='Location of Lackner nb1 dat file')
parser.add_argument('--nb4', '-s', metavar='/path/to/file.dat', default='',
                    type=str, help='Location of Lackner nb4 dat file')
parser.add_argument('--progress', '-P', action='store_true',
                    help='Whether to use a progress bar')

args = parser.parse_args()

sid_list = np.loadtxt('subject-id-list.csv', dtype='u8')
gzb_c_df = pd.DataFrame(
    gu.meta_map
).T[['ra', 'dec']].loc[sid_list].dropna().astype(float)

gzb_coordinates = SkyCoord(
    gzb_c_df.values * u.degree
)

if args.nb1:
    print('Making nb1')
    with open(args.nb1) as f:
        lackner_nb1_data = f.read()

    column_map1 = {
        int(k): v.strip()
        for k, v in (
            i.groups()
            for i in (
                re.search(r'# ?([0-9]+) +(.*?) ', i)
                for i in lackner_nb1_data.split('\n')
                )
            if i is not None
        )
    }

    lackner_nb1 = pd.read_csv(
        args.nb1,
        sep='\s+', comment='#', skiprows=0, header=None,
        names=[column_map1[i + 1] for i in range(len(column_map1.keys()))],
    )
    nb1_idexes, sep, _ = gzb_coordinates.match_to_catalog_sky(
      SkyCoord(lackner_nb1[['SDSS_RA', 'SDSS_DEC']].values * u.degree)
    )
    nb1_subset = lackner_nb1.iloc[nb1_idexes]
    nb1_subset.index = sid_list
    nb1_subset = nb1_subset[sep.arcsec < 10]
    nb1_subset.describe()
    nb1_subset.to_csv('lackner_nb1.csv')

if args.nb4:
    print('Making nb4')
    with open(args.nb4) as f:
        lackner_nb4_data = f.read()

    column_map4 = {
        int(k): v.strip()
        for k, v in (
            i.groups()
            for i in (
                re.search(r'# ?([0-9]+) +(.*?) ', i)
                for i in lackner_nb4_data.split('\n')
                )
            if i is not None
        )
    }
    cols4 = [column_map4[i + 1] for i in range(len(column_map4.keys()))]
    lackner_nb4 = pd.read_csv(
        args.nb4,
        sep='\s+', comment='#', skiprows=0,
        header=None, names=cols4,
    )
    nb4_idexes, sep, _ = gzb_coordinates.match_to_catalog_sky(
      SkyCoord(lackner_nb4[['SDSS_RA', 'SDSS_DEC']].values * u.degree)
    )
    nb4_subset = lackner_nb4.iloc[nb1_idexes]
    nb4_subset.index = sid_list
    nb4_subset = nb4_subset[sep.arcsec < 10]
    nb4_subset.describe()
    nb4_subset.to_csv('lackner_nb4.csv')


lackner_nb1.columns
