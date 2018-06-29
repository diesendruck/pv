import argparse
import numpy as np
import pandas as pd
from gmplot import gmplot


"""Plots lat-lon pairs on Google Maps.
   See for reference:
     https://github.com/vgm64/gmplot/blob/master/gmplot/gmplot.py
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_to_plot', type=int, default=40)
parser.add_argument('--plot_orig', default=False, action='store_true',
                    dest='plot_orig')
parser.add_argument('--do_reload', default=False, action='store_true',
                    dest='do_reload')
args = parser.parse_args()
num_to_plot = args.num_to_plot
plot_orig = args.plot_orig
do_reload = args.do_reload


if do_reload:
    rides_raw = pd.read_csv("rides.csv")
    rides = rides_raw[
        ['start_location_lat', 'start_location_long',
         'end_location_lat', 'end_location_long']]
    np.save('rides_lat_lon.npy', np.array(rides))

else:
    print('Loading rides.')
    if plot_orig:
        # rides = np.load('rides_lat_lon.npy')
        rides = np.load('data_train.npy')
    else:
        rides = np.load('g_out.npy')
        rides = np.load('g_out1000.npy')

# Format like this: ride = [(start_lat, start_lon), (end_lat, end_lon)]
formatted_rides_full = np.array([[tuple(r[:2]), tuple(r[2:])] for r in rides])

# Choose a random subset to plot.
# formatted_rides = formatted_rides_full[np.random.choice(
#     len(formatted_rides_full), num_to_plot)]
formatted_rides = formatted_rides_full[:num_to_plot]
print('Of {} total, plotting first {}.'.format(len(formatted_rides_full),
                                               num_to_plot))


# Place map
gmap = gmplot.GoogleMapPlotter(30.269805, -97.743203, 11)

# Polygon
print('Plotting {} rides.'.format(num_to_plot))
for ride in formatted_rides[:num_to_plot]:
    # Plot ride.
    ride_lats, ride_lons = zip(*ride)
    gmap.plot(ride_lats, ride_lons, 'cornflowerblue', edge_width=1)
    # Plot starting and ending points.
    start_lat, start_lon = ride[0]
    end_lat, end_lon = ride[1]
    gmap.scatter([start_lat], [start_lon], 'mediumaquamarine', marker=False,
                 face_alpha=0.6)
    gmap.scatter([end_lat], [end_lon], 'tomato', marker=False, face_alpha=0.6)

# Polygon
print('Drawing map.')
if plot_orig:
    gmap.draw('austin.html')
else:
    gmap.draw('austin_sim.html')

print('Done.\n')
