import json
import numpy as np
import pandas as pd
import lib.galaxy_utilities as gu
from pprint import pprint


def get_frac(mask):
    return '{:.2%} out of {}'.format(mask.sum() / mask.size, mask.size)


param_limits = {
    'disk': {
        'scale': (0, 2, 1),
        'i0': (0, 1, 0.2),
    },
    'bulge': {
        'scale': (0, 2, 1),
        'i0': (0, 2, 0.5),
        'n': (0.5, 5, 1),
    },
    'bar': {
        'scale': (0, 2, 1),
        'i0': (0, 1, 0.2),
        'n': (0.3, 2, 0.5),
        'c': (1.5, 3, 2),
    },
    'spiral': {
        'i0': (0, 1, 0.75),
        'spread': (0, 2, 1),
        'falloff': (0, 2, 1),
    }
}

rename_dict = dict(T13='intensity', T15='sersic', T16='intensity')

sid_list = np.loadtxt('lib/subject-id-list.csv', dtype='u8')

annotations = gu.classifications['annotations'][
    np.isin(gu.classifications['subject_ids'], sid_list)
].apply(json.loads)

annotations = annotations[annotations.apply(len) == 4]

drawn_disks = annotations.apply(lambda ann: (
    ann[0]['value'][0]['value'][0]
    if len(ann[0]['value'][0]['value']) > 0
    else np.nan
)).dropna().apply(pd.Series)

disk_sliders = annotations.apply(
    lambda ann: (
        {
            a['task'].replace('Slider', ''): a['value']
            for a in ann[0]['value'][1:]
        }
        if len(ann[0]['value'][0]['value']) > 0
        else np.nan
    )
).dropna().apply(pd.Series).rename(columns=rename_dict)

drawn_bulges = annotations.apply(lambda ann: (
    ann[1]['value'][0]['value'][0]
    if len(ann[1]['value'][0]['value']) > 0
    else np.nan
)).dropna().apply(pd.Series)

bulge_sliders = annotations.apply(
    lambda ann: (
        {
            a['task'].replace('Slider', ''): a['value']
            for a in ann[1]['value'][1:]
        }
        if len(ann[1]['value'][0]['value']) > 0
        else np.nan
    )
).dropna().apply(pd.Series).rename(columns=rename_dict)

drawn_bars = annotations.apply(lambda ann: (
    ann[2]['value'][0]['value'][0]
    if len(ann[2]['value'][0]['value']) > 0
    else np.nan
)).dropna().apply(pd.Series)

bar_sliders = annotations.apply(
    lambda ann: (
        {
            a['task'].replace('Slider', ''): a['value']
            for a in ann[2]['value'][1:]
        }
        if len(ann[2]['value'][0]['value']) > 0
        else np.nan
    )
).dropna().apply(pd.Series).rename(columns=rename_dict)


## How often do we see bad (default or extreme) values?
disk_axratio_is_bad = (drawn_disks.rx / drawn_disks.ry == 0.5) ^ (drawn_disks.rx / drawn_disks.ry == 2)
disk_scale_is_bad = disk_sliders.eval('scale == {} | scale == {} | scale == {}'.format(*param_limits['disk']['scale']))
disk_intensity_is_bad = disk_sliders.eval('scale == {} | scale == {} | scale == {}'.format(*param_limits['disk']['i0']))

bulge_axratio_is_bad = (drawn_bulges.rx / drawn_bulges.ry == 0.5) ^ (drawn_bulges.rx / drawn_bulges.ry == 2)
bulge_scale_is_bad = bulge_sliders.eval('scale == {} | scale == {} | scale == {}'.format(*param_limits['bulge']['scale']))
bulge_intensity_is_bad = bulge_sliders.eval('scale == {} | scale == {} | scale == {}'.format(*param_limits['bulge']['i0']))
bulge_sersic_is_bad = bulge_sliders.eval('scale == {} | scale == {} | scale == {}'.format(*param_limits['bulge']['n']))

bar_rotation_is_bad = drawn_bars.angle == 0
bar_scale_is_bad = bar_sliders.eval('scale == {} | scale == {} | scale == {}'.format(*param_limits['bar']['scale']))
bar_intensity_is_bad = bar_sliders.eval('scale == {} | scale == {} | scale == {}'.format(*param_limits['bar']['i0']))
bar_sersic_is_bad = bar_sliders.eval('scale == {} | scale == {} | scale == {}'.format(*param_limits['bar']['n']))
bar_boxyness_is_bad = bar_sliders.eval('scale == {} | scale == {} | scale == {}'.format(*param_limits['bar']['c']))

bad ={
    'disk': dict(
        axratio=get_frac(disk_axratio_is_bad),
        scale=get_frac(disk_scale_is_bad),
        i0=get_frac(disk_intensity_is_bad),
    ),
    'bulge': dict(
        axratio=get_frac(bulge_axratio_is_bad),
        scale=get_frac(bulge_scale_is_bad),
        i0=get_frac(bulge_intensity_is_bad),
        n=get_frac(bulge_sersic_is_bad),
    ),
    'bar': dict(
        angle=get_frac(bar_rotation_is_bad),
        scale=get_frac(bar_scale_is_bad),
        i0=get_frac(bar_intensity_is_bad),
        n=get_frac(bar_sersic_is_bad),
        c=get_frac(bar_boxyness_is_bad),
    ),
}

print('For all classifications:')
print('\n'.join('{} {}: {}'.format(k, k2, v) for k in ('disk', 'bulge', 'bar') for k2, v in bad[k].items()))

bi = pd.read_pickle('lib/best_individual.pickle')
bi_disk = bi.Model.apply(lambda m: m['disk']).dropna().apply(pd.Series)
bi_bulge = bi.Model.apply(lambda m: m['bulge']).dropna().apply(pd.Series)
bi_bar = bi.Model.apply(lambda m: m['bar']).dropna().apply(pd.Series)

bi_disk_axratio_is_bad = bi_disk.axRatio == 0.5
bi_disk_intensity_is_bad = bi_disk.eval('i0 == {} | i0 == {} | i0 == {}'.format(*param_limits['disk']['i0']))

bi_bulge_axratio_is_bad = bi_bulge.axRatio == 0.5
bi_bulge_intensity_is_bad = bi_bulge.eval('i0 == {} | i0 == {} | i0 == {}'.format(*param_limits['bulge']['i0']))
bi_bulge_sersic_is_bad = bi_bulge.eval('n == {} | n == {} | n == {}'.format(*param_limits['bulge']['n']))

bi_bar_rotation_is_bad = bi_bar.roll == 0
bi_bar_intensity_is_bad = bi_bar.eval('i0 == {} | i0 == {} | i0 == {}'.format(*param_limits['bar']['i0']))
bi_bar_sersic_is_bad = bi_bar.eval('n == {} | n == {} | n == {}'.format(*param_limits['bar']['n']))
bi_bar_boxyness_is_bad = bi_bar.eval('c == {} | c == {} | c == {}'.format(*param_limits['bar']['c']))

bi_bad ={
    'disk': dict(
        axratio=get_frac(bi_disk_axratio_is_bad),
        i0=get_frac(bi_disk_intensity_is_bad),
    ),
    'bulge': dict(
        axratio=get_frac(bi_bulge_axratio_is_bad),
        i0=get_frac(bi_bulge_intensity_is_bad),
        n=get_frac(bi_bulge_sersic_is_bad),
    ),
    'bar': dict(
        angle=get_frac(bi_bar_rotation_is_bad),
        i0=get_frac(bi_bar_intensity_is_bad),
        n=get_frac(bi_bar_sersic_is_bad),
        c=get_frac(bi_bar_boxyness_is_bad),
    ),
}
print('\nFor best individual classifications:')
print('\n'.join('{} {}: {}'.format(k, k2, v) for k in ('disk', 'bulge', 'bar') for k2, v in bi_bad[k].items()))
