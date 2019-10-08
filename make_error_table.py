import pandas as pd

agg_results = pd.read_pickle('lib/aggregation_results.pickle')
diff_data_df = pd.read_pickle('lib/fitting_metadata.pkl')

models = agg_results.Model.apply(pd.Series)
disk_scale = models.disk.apply(pd.Series)
bulge_scale = models.bulge.apply(pd.Series)
bar_scale = models.bar.apply(pd.Series)

disk_scale['axRatio'] = 1
bulge_scale['axRatio'] = 1
bar_scale['axRatio'] = 1


err = agg_results.Errors.apply(pd.Series)
cols = err.bar.apply(pd.Series).columns.values

# scale error relative to values

disk_err = err.disk.apply(pd.Series) / disk_scale
bulge_err = err.bulge.apply(pd.Series) / bulge_scale
bar_err = err.bar.apply(pd.Series) / bar_scale

err_df = pd.concat((
    disk_err.apply(pd.Series).rename(
        columns={k: 'disk-{}'.format(k) for k in cols}
    ).sort_index(),
    bulge_err.apply(pd.Series).rename(
        columns={k: 'bulge-{}'.format(k) for k in cols}
    ).sort_index(),
    bar_err.apply(pd.Series).rename(
        columns={k: 'bar-{}'.format(k) for k in cols}
    ).sort_index()
), axis=1, sort=True)

print(err_df.describe().round(2).T.dropna().to_latex())
