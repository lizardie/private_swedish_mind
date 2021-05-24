"""
The  main file for the package
==============================
"""

import geopandas as gpd
import contextily as ctx
from shapely.ops import unary_union
import pandas as pd
from shapely.geometry import Point
from geovoronoi import voronoi_regions_from_coords, points_to_coords

import matplotlib.pyplot  as plt
import numpy as np

import os

SOURCE_EPSG = 4326
WGS84_EPSG  = 3857
SWEREF_EPSG =  3006

G3_ANTENNAS_PATH = "antennas/UMTS.CSV.gz"
G4_ANTENNAS_PATH = "antennas/LTE.CSV.gz"

SWEREF_EPSG_uppsala =  (647742, 6638924.00)

# pd.set_option('display.max_colwidth', None)

SVERIGE_CONTOUR = 'sverige_contour/sverige.shp'
ALLA_KOMMUNER  = 'alla_kommuner/alla_kommuner.shp'

# dates  = ["2019-05-23", "2019-05-26", "2019-06-12", "2019-06-22", "2018-10-26_2018-10-27",
#          "2018-05-23_2018-05-24", "2017-11-20_2017-11-21"]

geometry_gps_csv = 'geometry_gps_csv'
geometry_mpn_csv = 'geometry_mpn_csv'
SE_SHAPEFILES = 'se_shapefiles/se_1km.shp'

DATA_DIR = 'data'


class AnalyseBasicJoinedData:
    """
    Class handles input  files with the schema: `['timestamp', 'geometry_gps_csv', 'geometry_mpn_csv']`,
    where `timestamp` is a timestamp, `geometry_gps_csv` is  string representation of a list of length one, which element is a position in a
    `WGS84_EPSG` projection. The `geometry_mpn_csv` is a string representation of a list  which element is a position in a
    `WGS84_EPSG` projection.
    """

    def __init__(self, point):
        """
        constructor
        """
        print("reading data")
        self.df = self.read_data()
        self.antennas_data = self.prepare_antennas()
        print('making bounding area')
        self.contour = self._get_bounding_area(point=point)

        print("antennas before filtering ",self.antennas_data.shape)
        print('filtering antennas data to be inside contour')
        self.antennas_data = self.get_objects_within_area(self.antennas_data)
        print("antennas after filtering ", self.antennas_data.shape)

        print('making voronoi polygons')
        self.vcs = self.create_voronoi_polygons().to_crs(WGS84_EPSG)

    @staticmethod
    def read_data():
        """
        reading data from the folder `DATA_DIR` which start with `result`.
        Returns Pandas DF, as a concatenation of all files

        :param dt: Data path
        :return: Pandas DF with `['timestamp', 'geometry_gps_csv', 'geometry_mpn_csv']`  columns
        """

        paths = sorted(
            [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.startswith('result')]
        )

        dfs = []

        for path in paths:
            try:
                tmp = pd.read_csv(path, parse_dates=['timestamp'])[
                    ['timestamp', 'geometry_gps_csv', 'geometry_mpn_csv']]
                print(tmp.shape[0], path)
                #         tmp = tmp.drop_duplicates(subset=['geometry_gps_csv'])
                #         print(tmp.shape[0], path)
                dfs.append(tmp)
            except IOError:
                print('path %s is wrong. check it.' % path)
                pass
            except KeyError:
                print("some keys are  missing... check it at: %s" % path)
                pass

        return pd.concat(dfs, axis=0, ignore_index=True)

    def process_df(self, ):
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
        self.df = self.df.groupby('timestamp').agg({geometry_gps_csv: "first", geometry_mpn_csv: list})

        self.df = gpd.GeoDataFrame(self.df, geometry=geometry_gps_csv, crs=WGS84_EPSG)  # .to_crs(SWEREF_EPSG)

        # return df



    def process_position_data(self):
        """

        :return:
        """
        print("processing data")
        self.process_df()
        print("processed data shape" ,self.df.shape)
        print('filtering  data to be inside contour')

        print("data before filtering:", self.df.shape)
        self.df = self.get_objects_within_area(self.df.to_crs(SWEREF_EPSG), geom='geometry_gps_csv').to_crs(WGS84_EPSG)
        print("data after filtering:", self.df.shape)

        print('adding add_vcs_indexes')
        self.add_vcs_indexes()
        print('adding rings column')
        self.add_rings_column()
        print("adding hist column")
        self.add_hist_column()



    @staticmethod
    def prepare_antennas():
        """
        reads and prepares antennas for whole Sweden.
        Returns a GeoPandas df with the antennas.
        """

        try:
            g3 = pd.read_csv(G3_ANTENNAS_PATH, sep=';')[['llat', 'llong']]
            g4 = pd.read_csv(G4_ANTENNAS_PATH, sep=';')[['llat', 'llong']]

            antennas = pd.concat([g3, g4]).drop_duplicates().round(3)

            antennas_gdp = gpd.GeoDataFrame(antennas, geometry=gpd.points_from_xy(antennas.llong, antennas.llat),
                                            crs=SOURCE_EPSG)[['geometry']] \
                .to_crs(SWEREF_EPSG)

            return antennas_gdp
        except:
            print("something is wrong with the antennas files")

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
            contour = gpd.read_file(SVERIGE_CONTOUR)
            contour.crs = WGS84_EPSG
            contour.to_crs(SWEREF_EPSG, inplace=True)

        return contour



    def _find_area(self, point):
        """
        given a point  within the area find that area in the table
        Returns geoPandas DF with the geometry
        """

        sweden = None
        try:
            sweden = gpd.read_file(ALLA_KOMMUNER)
            sweden.crs = SWEREF_EPSG
        except:
            print('failed to read %s' % ALLA_KOMMUNER)

        uppsala = Point(point)

        uppsala_lan = None
        for index, row in sweden.iterrows():
            if uppsala.within(sweden.iloc[index].geometry):
                uppsala_lan = sweden.iloc[index:index + 1].reset_index()

        if not uppsala_lan.empty:
            return uppsala_lan
        else:
            print('failed to find point %s in %s' % (point, ALLA_KOMMUNER))


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
                # if hasattr(self, objects):
                #     print(type(objects))
                # else:
                #     print("dfsffd")
                return gpd.GeoDataFrame(objects_within, geometry=geom, crs=objects.crs).reset_index()
            else:
                print("no objects found within area")

            # return objects_within
        else:
            print('Objects have different CRSs: %s and %s ' % (objects.crs, self.contour.crs))
            # return None


    def create_voronoi_polygons(self):
        """
        creates Voronoi polygons with antennas as centers, bounded by `bounding_geometry`

        """
        coords = points_to_coords(self.antennas_data.geometry)
        print(self.antennas_data.crs, self.contour.crs)
        print(type(self.contour.geometry[0]))
        poly_shapes, pts = voronoi_regions_from_coords(coords, self.contour.geometry[0])

        voronoi_polygons = gpd.GeoDataFrame({'geometry': poly_shapes}, crs=SWEREF_EPSG)

        return voronoi_polygons



    def add_vcs_indexes(self):
        """
        Adding indexes of Voronoi cell from `vcs` for GPS and MPN points for given `df`
        """

        if self.df.crs == self.vcs.crs:

            vc_gps_points = []
            vc_mpn_points = []

            #     temp = vcs.to_crs(WGS84_EPSG).geometry
            temp = self.vcs.geometry

            for point in self.df.geometry_gps_csv:
                for i, vc in enumerate(temp):
                    if (point.within(vc)): vc_gps_points.append(i)

            for points in self.df.geometry_mpn_csv:
                t = []
                for point in points:
                    for i, vc in enumerate(temp):
                        if (point.within(vc)): t.append(i)
                vc_mpn_points.append(t)
            print(self.df.shape, len(vc_gps_points))
            self.df['vc_index_gps'] = vc_gps_points
            self.df['vc_index_mpn'] = vc_mpn_points

            # return df
        else:
            print('Objects have different CRSs: %s and %s '%(self.df.crs, self.vcs.crs))
            # return None



    def add_rings_column(self):
        """
        constructing rings column
        :return:
        """
        tmp = self.vcs.to_crs(WGS84_EPSG)

        self.df['vc_gps_rings'] = self.df.apply(lambda row: get_rings_around_cell(row['vc_index_gps'], tmp, 6), axis=1)


    def add_hist_column(self):
        """
        adding histogram column
        :return:
        """

        self.df['hist'] = self.df.apply(lambda row: make_hist_mpn_geoms(row['vc_index_mpn'], row['vc_gps_rings']),
                                        axis=1)


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


def make_hist_mpn_geoms(mpn_geoms, cell_rings):
    """
    taking list of `mpn_geoms` and corresponding list of cell rings `cell_rings`.
    Makes histogram for polulation of cell rings based in `mpn_geoms`.


    """

    hist = []
    for idx, ring in enumerate(cell_rings):
        for geom in mpn_geoms:
            if (geom in ring):
                hist.append(idx)
    return hist







def _plot_ring_hist(data):
    """
    plotting histogram for rings
    """

    hst = [el for sublist in data.to_list() for el in sublist]
    his = np.histogram(hst,bins=range(max(hst)+2))


    plt.figure(figsize=(15,10))
    plt.xlabel('ring number')
    plt.ylabel('number of occurences')
    plt.title("""Voronoi cell ring histogram for a GPS position
    averaged over 5 minute intervals. Number of GPS positions is %i."""%(read_df.shape[0]))
    plt.bar(his[1][0:-1],his[0])


    plt.savefig('hist.png')




if __name__ == "__main__":

    print("hello")

    if not os.path.isfile(SVERIGE_CONTOUR):
        print(SVERIGE_CONTOUR + " does  not exist... creating")
        t = gpd.read_file(SE_SHAPEFILES)
        t.to_crs(WGS84_EPSG, inplace=True)
        g = t.geometry.unary_union
        gpd.GeoDataFrame(geometry=gpd.GeoSeries(g)).to_file(SVERIGE_CONTOUR)


    data = AnalyseBasicJoinedData(point=SWEREF_EPSG_uppsala)

    data.process_position_data()

    # antennas_gdp = prepare_antennas()
    #
    # # sweden = get_bounding_area()
    # uppsala_lan = get_bounding_area(point=SWEREF_EPSG_uppsala)
    #
    #
    # # antennas_within, voronoi_polygons = get_vcs_for_bounding_area(antennas_gdp, sweden)
    # antennas_within, voronoi_polygons = get_vcs_for_bounding_area(antennas_gdp, uppsala_lan)
    #
    # read_df = read_data()
    # read_df = process_df(read_df.copy())
    # print(type(uppsala_lan), type(read_df))
    # read_df = get_objects_within_area(read_df.copy(), uppsala_lan.to_crs(WGS84_EPSG), geom='geometry_gps_csv')
    #
    # read_df = add_vcs_indexes(read_df.copy(), voronoi_polygons.to_crs(WGS84_EPSG))
    #
    # tmp = voronoi_polygons.to_crs(WGS84_EPSG)
    #
    # read_df['vc_gps_rings'] = read_df.apply(lambda row: get_rings_around_cell(row['vc_index_gps'], tmp, 6), axis=1)
    #
    # read_df['hist'] = read_df.apply(lambda row: make_hist_mpn_geoms(row['vc_index_mpn'], row['vc_gps_rings']), axis=1)
    #
    # _plot_ring_hist(read_df['hist'])

    print('done')