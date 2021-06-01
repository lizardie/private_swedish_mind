"""
The  main file for the package
==============================
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from geovoronoi import voronoi_regions_from_coords, points_to_coords

import matplotlib.pyplot  as plt
import numpy as np

import os

import logging
import collections


try:
    from private_swedish_mind.utils import make_hist_mpn_geoms, get_rings_around_cell,make_group_col,get_vcs_used_area,\
        get_splits, make_group_load_col
    from private_swedish_mind.consts import  *
    import private_swedish_mind.plots

except ModuleNotFoundError:
    from utils import make_hist_mpn_geoms, get_rings_around_cell, make_group_col, get_vcs_used_area,\
        get_splits, make_group_load_col
    from consts import *
    import plots





# import logging.handlers

# SOURCE_EPSG = 4326
# WGS84_EPSG  = 3857
# SWEREF_EPSG =  3006

G3_ANTENNAS_PATH = "antennas/UMTS.CSV.gz"
G4_ANTENNAS_PATH = "antennas/LTE.CSV.gz"

SWEREF_EPSG_uppsala =  (647742, 6638924)

# pd.set_option('display.max_colwidth', None)

SVERIGE_CONTOUR = 'sverige_contour/sverige.shp'
ALLA_KOMMUNER  = 'alla_kommuner/alla_kommuner.shp'

# column names of the DF
geometry_gps_csv = 'geometry_gps_csv'
geometry_mpn_csv = 'geometry_mpn_csv'
vc_index_gps = 'vc_index_gps'
vc_index_mpn = 'vc_index_mpn'
vc_gps_rings = 'vc_gps_rings'
hist = 'hist'
timestamp = 'timestamp'

# columns in antennas files
lat = 'llat'
long = 'llong'


SE_SHAPEFILES = 'se_shapefiles/se_1km.shp'

DATA_DIR = 'data'
ANTENNAS_LOAD_DIR = 'data/antennas_load'




class AnalyseBasicJoinedData:
    """
    Class handles input  files with the schema: `['timestamp', 'geometry_gps_csv', 'geometry_mpn_csv']`,
    where `timestamp` is a timestamp, `geometry_gps_csv` is  string representation of a list of length one, which element is a position in a
    `WGS84_EPSG` projection. The `geometry_mpn_csv` is a string representation of a list  which element is a position in a
    `WGS84_EPSG` projection.
    """

    def __init__(self, point, n_layers):
        """
        creates object

        :param point: tuple like (647742, 6638924) describing a point within Lan, in  SWEREF_EPSG
        :param n_layers: number of layers of Voronoi cells to build
        """

        logger.info("reading data")
        self.df = self.read_data()
        logger.info("reading antennas")
        self.antennas_data = self.read_antennas()
        logger.info('making bounding area')
        self.contour = self._get_bounding_area(point=point)
        self.vcs = None
        self.n_layers = n_layers
        self.most_used_antennas = None
        logger.info("reading antennas load files")
        self.antennas_load = self.read_antennas_load()


    @staticmethod
    def read_data():
        """
        reading data from the folder `DATA_DIR` which start with `result`.
        Returns Pandas DF, as a concatenation of all files

        :return: Pandas DF with `['timestamp', 'geometry_gps_csv', 'geometry_mpn_csv']`  columns
        """

        paths = sorted(
            [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.startswith('result')]
        )

        dfs = []

        for path in paths:
            try:
                tmp = pd.read_csv(path, parse_dates=[timestamp])[
                    [timestamp, geometry_gps_csv, geometry_mpn_csv]]
                logger.info("number of lines: %s, location %s " %(tmp.shape[0], path))
                #         tmp = tmp.drop_duplicates(subset=['geometry_gps_csv'])
                #         print(tmp.shape[0], path)
                dfs.append(tmp)
            except IOError:
                logger.error('path %s is wrong. check it.' % path)
                pass
            except KeyError:
                logger.warning("some keys are  missing... check it at: %s" % path)
                pass

        return pd.concat(dfs, axis=0, ignore_index=True)


    @staticmethod
    def read_antennas_load():
        """
        reading data for given dates
        """
        paths = sorted([os.path.join(ANTENNAS_LOAD_DIR, f) for f in os.listdir(ANTENNAS_LOAD_DIR)])

        dfs = []

        for path in paths:
            try:
                tmp = pd.read_csv(path)
                logger.info("number of lines: %s, location %s " %(tmp.shape[0], path))
                dfs.append(tmp)
            except IOError:
                logger.error('path %s is wrong. check it.' % path)
                pass
            except KeyError:
                logger.warning("some keys are  missing... check it at: %s" % path)
                pass

        return pd.concat(dfs, axis=0, ignore_index=True)



    def transform_df(self, ):
        """
        transforms string representation of a list to a Shapely point, and
        groups the data for each timestamp.

        :return: GeoPandas DF with `timestamp`, Point(), [Point(), Point(), ... Point()] schema
        """
        df = self.df

        self.df[geometry_gps_csv] = self.df[geometry_gps_csv].apply(
            lambda lst: Point(float(lst[1:-1].split(',')[0]),
                              float(lst[1:-1].split(',')[1]))
        )
        self.df[geometry_mpn_csv] = self.df[geometry_mpn_csv].apply(
            lambda lst: Point(float(lst[1:-1].split(',')[0]),
                              float(lst[1:-1].split(',')[1]))
        )
        self.df = self.df.groupby(timestamp).agg({geometry_gps_csv: "first", geometry_mpn_csv: list})

        self.df = gpd.GeoDataFrame(self.df, geometry=geometry_gps_csv, crs=WGS84_EPSG)  # .to_crs(SWEREF_EPSG)

        # return df



    def remove_too_often_antennas(self):
        """

        :return:
        """
        most_common_antennas = collections.Counter(self.df[geometry_mpn_csv]).most_common(50)
        # antennas[0].__str__()
        # print(most_common_antennas)
        most_common_antennas_name = [p[0] for p in most_common_antennas]
        self.most_used_antennas = most_common_antennas
        self.df = self.df[
            ~self.df[geometry_mpn_csv].isin(most_common_antennas_name[:6])
        ]



    def process_position_data(self):
        """

        :return:
        """


        logger.info("before removing too often used antennas: %s" %str(self.df.shape))
        self.remove_too_often_antennas()
        logger.info("after removing too often used antennas: %s" %str(self.df.shape))


        logger.info("antennas before filtering %s" %str(self.antennas_data.shape))

        logger.info('filtering antennas data to be inside contour')
        self.antennas_data = self.get_objects_within_area(self.antennas_data)
        logger.info("antennas after filtering: %s " % str(self.antennas_data.shape))

        logger.info('making voronoi polygons')
        self.vcs = self.create_voronoi_polygons().to_crs(WGS84_EPSG)

        logger.info("processing data")
        self.transform_df()
        logger.info("processed data shape: %s" % str(self.df.shape))
        logger.info('filtering  data to be inside contour')

        logger.info("data before filtering: %s" %str(self.df.shape))
        self.df = self.get_objects_within_area(self.df.to_crs(SWEREF_EPSG), geom=geometry_gps_csv).to_crs(WGS84_EPSG)
        logger.info("data after filtering: %s" %str(self.df.shape))

        logger.info('adding add_vcs_indexes')
        self.add_vcs_indexes()
        logger.info('adding rings column')
        self.add_rings_column()
        logger.info("adding hist column")
        self.add_hist_column()
        logger.info("adding hist groups...")
        self.add_hist_groups_column()
        logger.info("adding distance column")
        self.add_distance_column()

        logger.info("processing antennas load")
        self.antennas_load = self.process_antennas_load()
        logger.info('done')
        logger.info("adding loads columns")
        self.add_hist_load_group_columns()


        return self.df


    def process_antennas_load(self, coarse_grain_factor=100):
        """
        processing coords and summing up load for each position
        """
        self.antennas_load['X'] = coarse_grain_factor * ((self.antennas_load['avg_X'] / coarse_grain_factor).astype('int'))
        self.antennas_load['Y'] = coarse_grain_factor * ((self.antennas_load['avg_Y'] / coarse_grain_factor).astype('int'))

        self.antennas_load = self.antennas_load.groupby(['X', 'Y']).agg({'num_ids_list': 'sum', 'num_ids_set': 'sum'}) \
            .reset_index()

        self.antennas_data['X'] = self.antennas_data['geometry'].apply(
            lambda val: coarse_grain_factor * int(val.x / coarse_grain_factor))
        self.antennas_data['Y'] = self.antennas_data['geometry'].apply(
            lambda val: coarse_grain_factor * int(val.y / coarse_grain_factor))

        antennas_within_loads = pd.merge(self.antennas_load, self.antennas_data, how='right', left_on=['X', 'Y'],
                                         right_on=['X', 'Y'])

        return antennas_within_loads



    @staticmethod
    def read_antennas():
        """
        reads and prepares antennas for whole Sweden.
        Returns a GeoPandas df with the antennas.
        """

        try:
            g3 = pd.read_csv(G3_ANTENNAS_PATH, sep=';')[[lat, long]]
            g4 = pd.read_csv(G4_ANTENNAS_PATH, sep=';')[[lat, long]]

            antennas = pd.concat([g3, g4]).round(3).drop_duplicates()

            antennas_gdp = gpd.GeoDataFrame(antennas, geometry=gpd.points_from_xy(antennas.llong, antennas.llat),
                                            crs=SOURCE_EPSG)[['geometry']] \
                .to_crs(SWEREF_EPSG)

            return antennas_gdp
        except IOError:
            logger.error("something is wrong with the antennas files: %s, %s"%(G3_ANTENNAS_PATH, G4_ANTENNAS_PATH))

            return None




    def _get_bounding_area(self, point=None):
        """
        returns bounding area.
        It could be either the whole Sweden, or it's Lan. In the last case one  need to provide a point within that lan in
        Sweref99 format.

        """
        if point:
            contour = self._find_area(point)
        else:
            try:
                contour = gpd.read_file(SVERIGE_CONTOUR)
                contour.crs = WGS84_EPSG
                contour.to_crs(SWEREF_EPSG, inplace=True)
            except IOError:
                logger.error('error reading contour file: %s ' %SVERIGE_CONTOUR)

        return contour



    def _find_area(self, point):
        """
        given a point  within the area find that area in the table.
        Returns geoPandas DF with the geometry

        :param point: tuple representing point within Sverige Lan
        :return: GeoPandas DF with the Lan for the Point
        """


        sweden = None
        try:
            sweden = gpd.read_file(ALLA_KOMMUNER)
            sweden.crs = SWEREF_EPSG
        except IOError:
            logger.error('failed to read %s' % ALLA_KOMMUNER)

        uppsala = Point(point)

        uppsala_lan = None
        for index, row in sweden.iterrows():
            if uppsala.within(sweden.iloc[index].geometry):
                uppsala_lan = sweden.iloc[index:index + 1].reset_index()

        if not uppsala_lan.empty:
            return uppsala_lan
        else:
            logger.error('failed to find point %s in %s' % (point, ALLA_KOMMUNER))


    # def get_vcs_for_bounding_area(self):
    #     """
    #     returns Voronoi cells and their centers
    #     """
    #     if self.contour.crs == self.antennas_data.crs:
    #         self.get_objects_within_area()
    #         voronoi_polygons = self.create_voronoi_polygons()
    #
    #         return antennas_within, voronoi_polygons
    #     else:
    #         print("objects have differrent CRSs....")
    #         return None, None


    def get_objects_within_area(self, objects, geom='geometry'):
        """
        given objects and bounding geometry find which objects are within the geometry
        Returns a GeoPandas DF
        """
        if objects.crs == self.contour.crs:
            objects_within = []
            for i, row in objects.iterrows():
                if row[geom].within(self.contour.geometry[0]):
                    objects_within.append(row)

            if objects_within:

                return gpd.GeoDataFrame(objects_within, geometry=geom, crs=objects.crs).reset_index(drop=True)
            else:
                logger.error("no objects found within area! ")

        else:
            logger.error('Objects have different CRSs: %s and %s ' % (objects.crs, self.contour.crs))
            # return None


    def create_voronoi_polygons(self):
        """
        creates Voronoi polygons with `self.antennas_data` as centers, bounded by `self.contour`

        :return: GeoPandas DF with VCs
        """

        if self.antennas_data.crs == self.contour.crs:

            coords = points_to_coords(self.antennas_data.geometry)
            poly_shapes, pts = voronoi_regions_from_coords(coords, self.contour.geometry[0])

            voronoi_polygons = gpd.GeoDataFrame({'geometry': poly_shapes}, crs=SWEREF_EPSG)

            return voronoi_polygons

        else:
            logger.error('Objects have different CRSs: %s and %s ' % (self.antennas_data.crs, self.contour.crs))




    def add_vcs_indexes(self):
        """
        Adding indexes of Voronoi cell from `vcs` for GPS and MPN points for given `df`
        """

        if self.df.crs == self.vcs.crs:

            vc_gps_points = []
            vc_mpn_points = []

            temp = self.vcs.geometry

            for point in self.df.geometry_gps_csv:
                for i, vc in enumerate(temp):
                    if point.within(vc): vc_gps_points.append(i)

            for points in self.df.geometry_mpn_csv:
                t = []
                for point in points:
                    for i, vc in enumerate(temp):
                        if point.within(vc): t.append(i)
                vc_mpn_points.append(t)
            # print(self.df.shape, len(vc_gps_points))
            self.df[vc_index_gps] = vc_gps_points
            self.df[vc_index_mpn] = vc_mpn_points

            # return df
        else:
            logger.error('Objects have different CRSs: %s and %s '%(self.df.crs, self.vcs.crs))
            # return None



    def add_rings_column(self):
        """
        constructing rings column `vc_gps_rings` centered around `vc_index_gps`.
        The number of layers is given by `self.n_layers`.


        :return: `vc_gps_rings` column to `self.df`
        """
        tmp = self.vcs.to_crs(WGS84_EPSG)
        # vc_used, area = get_vcs_used_area(tmp, self.df[[vc_index_gps, vc_index_mpn]], area_max=200)

        self.df[vc_gps_rings] = self.df.apply(lambda row: get_rings_around_cell(row[vc_index_gps], tmp, self.n_layers), axis=1)


    def add_hist_column(self):
        """
        adding histogram by looping through mnp-indexes and vc-indexes

        :return: column `hist`  to `self.df`
        """

        self.df[hist] = self.df.apply(lambda row: make_hist_mpn_geoms(row[vc_index_mpn], row[vc_gps_rings]),
                                        axis=1)



    def add_hist_groups_column(self):
        """
        adding histograms for different groups of VC sizes

        :return:
        """
        tmp = self.vcs.to_crs(WGS84_EPSG)
        vc_used, area = get_vcs_used_area(tmp, self.df[[vc_index_gps, vc_index_mpn]], area_max=200)

        size_borders = get_splits(area, 3, make_int=False)

        for key, size_ in enumerate(size_borders):
            self.df['group' + str(key)] = self.df.apply(lambda row:
                                                        make_group_col(row[vc_index_mpn], tmp, size_),
                                                        axis=1)

            self.df[hist + str(key)] = self.df[[hist, 'group' + str(key)]].apply(
                lambda row: [el[0] for el in zip(row[hist], row['group' + str(key)]) if el[1] == True],
                axis=1)



    def add_hist_load_group_columns(self):
        """
        adding columns for different load groups

        :return:
        """
        tmp = gpd.GeoDataFrame(self.antennas_load, crs=SWEREF_EPSG).to_crs(WGS84_EPSG)

        logger.info("creating splits for antennas load")
        load_borders = get_splits(self.antennas_load['num_ids_list'].dropna(), 3, make_int=True)
        logger.info("done")


        for key, size_ in enumerate(load_borders):
            self.df['group_load' + str(key)] = self.df.apply(lambda row:
                                                             make_group_load_col(row['vc_index_mpn'], tmp, size_),
                                                             axis=1)

            self.df['hist_load' + str(key)] = self.df[['hist', 'group_load' + str(key)]].apply(
                lambda row: [el[0] for el in zip(row['hist'], row['group_load' + str(key)]) if el[1] == True],
                axis=1)



    def add_distance_column(self):
        """
        adding distance column.


        :return: list of distances  between GPS and each of MPN positions
        """
        self.df['distances'] = self.df.apply(lambda row: [row[geometry_gps_csv].distance(item) / 1000 \
                                                      for item in row[geometry_mpn_csv]], axis=1)


    def make_plots(self):
        """
        producing all the plots

        :return:
        """

        logger.info('plotting  distance histogram')
        plots._plot_dist_hist(self.df['distances'], bins1=20, bins2=2, saveas="hist_dist.png")

        logger.info("plotting ring histogram")
        plots._plot_ring_hist(self.df['hist'], density=False, saveas="hist_ring.png")

        logger.info('plotting map and all visited VCs')
        vc_used, area = get_vcs_used_area(self.vcs, self.df[[vc_index_gps, vc_index_mpn]], area_max=200)
        plots._plot_visited_vcs_sizes(vc_used, area, area_max=200, saveas="visited_vcs_sizes.png")

        logger.info('plotting antennas usage histogram')
        plots._plot_antennas_usage(self.most_used_antennas, nbins=50, saveas="most_used_antennas.png")


        logger.info('plotting ring histogram by group')
        size_borders = get_splits(area, 3, make_int=False)
        plots._plot_ring_histogram_by_group(self.df[['hist', 'hist0', 'hist1', 'hist2']], size_borders, title=
        """Voronoi cell ring histogram for a GPS position. Number of MPN positions is %i,  number of timestamps is %i.
        Plots are for different size-based classes of VCs""",
                                      label="area is within  %s $km^2$", saveas='hist_ring_by_group.png')


        logger.info('plotting ring histogram by load group')
        load_borders = get_splits(self.antennas_load['num_ids_list'].dropna(), 3, make_int=True)
        plots._plot_ring_histogram_by_group(self.df[['hist', 'hist_load0', 'hist_load1', 'hist_load2']], load_borders, title=
        """Voronoi cell ring histogram for a GPS position. Number of MPN positions is %i,  number of timestamps is %i.
        Plots are for different antennas load-based classes of VCs""",
                                      label="load  is within  %s connections", saveas='hist_ring_by_load_group.png')



        #
        # series_length = 10
        # series_diffs_pd = make_diffs_ring_histogram_sample_size(read_df['hist'], series_length, saveas="hist_ring_diffs_sample_size.png")
        #
        # _plot_ring_hist_diffs_sample_size(series_diffs_pd)


if __name__ == "__main__":


    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # handler = logging.handlers.SysLogHandler(address='/dev/log')
    # handler = logging.handlers.SysLogHandler()
    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('log.log')

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s [%(module)s.%(funcName)s:%(lineno)d] : %(message)s')
    s_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)


    logger.info("hello")

    if not os.path.isfile(SVERIGE_CONTOUR):
        logger.info(SVERIGE_CONTOUR + " does  not exist... creating")
        t = gpd.read_file(SE_SHAPEFILES)
        t.to_crs(WGS84_EPSG, inplace=True)
        g = t.geometry.unary_union
        gpd.GeoDataFrame(geometry=gpd.GeoSeries(g)).to_file(SVERIGE_CONTOUR)


    data = AnalyseBasicJoinedData(point=SWEREF_EPSG_uppsala, n_layers=6)

    print(data.process_position_data())

    logger.info('making plots')

    data.make_plots()


    logger.info('done')