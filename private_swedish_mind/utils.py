"""
Utilities
=========

Auxilliary routines for the main module
"""

import  pandas as pd
from shapely.ops import unary_union
import  numpy as np



def make_hist_mpn_geoms(mpn_geoms, cell_rings):
    """
    taking list of `mpn_geoms` and corresponding list of cell rings `cell_rings`.
    Makes histogram for population of cell rings based in `mpn_geoms`.


    """

    hist = []
    for idx, ring in enumerate(cell_rings):
        for geom in mpn_geoms:
            if (geom in ring):
                hist.append(idx)
    return hist


def get_rings_around_cell(cell_idx, vcs, width):
    """
    given the Voronoi cells `vcs`, the index of the cell `cell_idx` and the number of
    layers `width`
    creates a list of lists, where the first list contains indexes  of `vcs` for the zero layer,
    the second -- for the first layer etc.
    """

    result = [vcs.iloc[cell_idx:cell_idx + 1]]
    t = [[cell_idx]]
    neighbours_dis = result[0].iloc[0].geometry
    for layer in range(width):
        neighbours = vcs[vcs.geometry.touches(neighbours_dis)]
        t.append(neighbours.index.to_list())
        result.append(neighbours)
        neighbours = pd.concat(result)
        neighbours_dis = unary_union(neighbours.geometry)

    return t


def get_vcs_used_area(vcs, data, area_max):
    """
    collecting Voronoi cells visited during all journeys and their area, if it below the given threshould
    """
    unique_vc_indexes_tracks = np.unique(data.explode('vc_index_mpn').dropna().values.flatten()
                                         )
    # print(unique_vc_indexes_tracks)
    vc_used = vcs.iloc[unique_vc_indexes_tracks]

    area = vc_used.geometry.area / 10**6
    area = area[area < area_max]

    return vc_used, area


# def vc_area_splits(area, n_parts):
#     """
#     given the Pandas Series with Voronoi cell areas, sorts the areas by accending order
#     and splits into given number of parts.
#
#     Returns a list of tuples with splits borders, like `[(0.01, 2.63), (2.71, 24.67), (25.16, 184.09)]`
#     """
#
#     splits = np.array_split(area.sort_values().round(2), n_parts)
#     size_borders = [(el.iloc[0], el.iloc[-1]) for el in splits]
#
#     return size_borders


def get_splits(load, n_parts, make_int=True):
    """
    given the Pandas Series with Voronoi cell areas, sorts the areas by accending order
    and splits into given number of parts.

    Returns a list of tuples with splits borders, like `[(0.01, 2.63), (2.71, 24.67), (25.16, 184.09)]`
    """

    splits = np.array_split(load.sort_values().round(2), n_parts)
    if make_int:
        size_borders = [(int(el.iloc[0]), int(el.iloc[-1])) for el in splits]
    else:
        size_borders = [(el.iloc[0], el.iloc[-1]) for el in splits]

    return size_borders


def make_group_load_col(row, vcs, size_):
    vc_idxs = [el for el in row]
    #     areas = vcs.iloc[vc_idxs].geometry.area/10**6
    loads_idxs = vcs.iloc[vc_idxs].num_ids_list

    return [(el < size_[1]) & (el > size_[0]) for el in loads_idxs]



def make_group_col(row, vcs, size_):
    vc_idxs = [el for el in row]
    areas = vcs.iloc[vc_idxs].geometry.area / 10**6

    return [(el < size_[1]) & (el > size_[0]) for el in areas]



def make_diffs_ring_histogram_sample_size(hist_data, series_length):
    """
    we take samples of different size from `hist` column, make a histogram for each sample and
    observe how the difference between it and the  histogram for full `hist` column.
    We learn how the difference evolves with sample size.

    returns Pandas DF with the differences
    """

    series_diffs = []
    series_length = series_length
    sample_size = [10, 50, 100, 200, 300, 500, len(hist_data)]
    n_bins = 7

    for serie in range(series_length):
        hist_vals = []
        for sample in sample_size:
            data = hist_data.sample(sample)
            hst = [el for sublist in data.to_list() for el in sublist]
            his = np.histogram(hst, bins=range(n_bins), density=True)
            hist_vals.append(his[0])

        diffs = [sum(abs(element - hist_vals[-1])) for element in hist_vals]
        series_diffs.append(diffs)

    series_diffs_pd = pd.DataFrame(series_diffs, columns=sample_size)
    return series_diffs_pd



