Modules
=======


The code takes CSV files located in the `private_swedish_mind/data/`.
Each file represents data for a journey.

The files have the following format:

+-----+---------------------+-----------------------------------------+----------------------------------------+
| idx | timestamp           |                   geometry_gps_csv      |                    geometry_mpn_csv    |
+=====+=====================+=========================================+========================================+
| 0   | 2019-05-23 13:50:00 | [1541218.3500328728, 8117969.018622401] | [1546361.310684306, 8122021.062659311] |
+-----+---------------------+-----------------------------------------+----------------------------------------+
| 1   | 2019-05-23 14:30:00 | [1542727.347574737, 8114445.220161615]  |	[1542862.1674228888, 8117797.551964085]|
+-----+---------------------+-----------------------------------------+----------------------------------------+
| ... |        ...          |            ...        	              |                 ...                    |
+-----+---------------------+-----------------------------------------+----------------------------------------+
|1048 |	2017-11-21 22:35:00 | [2090807.2508528575, 8330158.61294096]  | [2088342.5148085796, 8331062.918022333]|
+-----+---------------------+-----------------------------------------+----------------------------------------+


where  `geometry_gps_csv` and `geometry_mpn_csv` are  the coordinates according  to GPS and MPN correspondingly in the `EPSG=3857` projection.


Next, data is undergoing the following transformation:

* the  data are being cut by the given bounding area
* the bounding area is being tesselated by Voronoi polygons with antennas as their centers

.. image:: pics/track_within_shape.png
    :width: 600
    :alt: Alternative text

* for each GPS position, `geometry_gps_csv` we identify to which Voronoi cell it  belongs to and consider it  as a base cell
* for each base cell we build the rings of Voronoi cells, as the year rings for the tree.  In the picture below the blue color shows the base cell, while the 'green', 'cyan', 'red', 'black' define the colors for  the rings of the next level.

.. image:: pics/upp_vor_cell.png
    :width: 600
    :alt: Alternative text

* for each `geometry_gps_csv` we build cell rings and for corresponding `geometry_mpn_csv` figure out to which ring **layer** it falls into.  Based on that we build a histogram for the layer  occurancies after moving  through the table.

.. image:: pics/hist_ring.png
    :width: 600
    :alt: Alternative text

The number for `ring=0` tells the number of times when the GPS and MPN positions fall into the same Voronoi cell.
The number for `ring=0` shows the number of times when MPN position is in the first layer, and so forth.



.. automodule:: private_swedish_mind.rings
   :members:
