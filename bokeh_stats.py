import sys
import pandas as pd
import numpy as np
from pathlib import Path
import re

import utils

from bokeh.io import output_notebook, show
# output_notebook()

from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, HoverTool, Span, Label, \
    LassoSelectTool, Select, LinearColorMapper, RangeSlider
from bokeh.models.glyphs import Circle
from bokeh.plotting import curdoc, figure, save
from bokeh.palettes import Blues, Greys256
from bokeh.colors import RGB

from decimal import Decimal


def fexp(number):
    sign, digits, exponent = Decimal(number).as_tuple()
    return len(digits) + exponent - 1


def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()


date = sys.argv[1]
home_folder = Path(f"D:/BJ_Data/{date}")

test = [(255, 255, 255), (255, 235, 235), (255, 215, 215), (255, 196, 196), (245, 179, 174), (255, 158, 158),
        (255, 124, 124), (255, 90, 90), (238, 80, 78), (244, 117, 75), (255, 160, 69), (255, 189, 87), (247, 215, 104),
        (240, 236, 121), (223, 245, 141), (205, 255, 162), (172, 245, 168), (138, 236, 174), (124, 235, 200),
        (106, 235, 225), (97, 225, 240), (68, 202, 255), (50, 190, 255), (25, 175, 255), (13, 129, 248), (26, 102, 240),
        (0, 40, 224), (0, 25, 212), (0, 10, 200), (20, 5, 175), (40, 0, 150), (10, 0, 121)]


def convert_to_hex(arr):
    return '#%02x%02x%02x' % arr


def load_conductance_data(direction, file_index):
    return pd.read_csv(home_folder.joinpath(f"results/conductance_stats_{direction}_{file_index}.csv"))


def load_noise_data(direction, file_index):
    return pd.read_csv(home_folder.joinpath(f"results/noise_stats_{direction}_{file_index}.csv"))


def calc_2d_hist(x_values, y_values, xrange, log_scale_x=False,
                 yrange=(1e-7, 10), num_bins=(100, 100)):
    if log_scale_x:
        num_of_decs = np.log10(xrange[1]) - np.log10(xrange[0])
        xbins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), num=int(num_bins[0] * num_of_decs), base=10)
    else:
        xbins = np.linspace(xrange[0], xrange[1], num=num_bins[0])

    num_of_decs = np.log10(yrange[1]) - np.log10(yrange[0])
    ybins = np.logspace(np.log10(yrange[0]), np.log10(yrange[1]), num=int(num_bins[1] * num_of_decs), base=10)

    h, xedges, yedges = np.histogram2d(x_values, y_values, bins=[xbins, ybins])
    x_mesh, y_mesh = np.meshgrid(xedges, yedges)
    xbins_arr = np.array([np.mean(xedges[i:i + 2]) for i in range(xedges.shape[0] - 1)])
    ybins_arr = np.array([np.mean(yedges[i:i + 2]) for i in range(yedges.shape[0] - 1)])

    x_width = np.diff(xedges)
    y_width = np.diff(yedges)

    # xbins_arr = xedges[:-1]
    # ybins_arr = yedges[:-1]

    return xbins_arr, ybins_arr, x_width, y_width, h.T


file_names = np.array(list(map(utils.get_name_from_path, home_folder.joinpath('results').glob(r'*.csv'))))
file_inds = np.array(list(map(utils.get_num_from_name, file_names)))
file_inds_un = np.unique(file_inds)

file_index_sel = Select(title="File index", options=list(file_inds_un.astype(str)), value='1')
direction_sel = Select(title="Direction", options=['pull', 'push'], value='pull')

conductance_stat = load_conductance_data(direction_sel.value, file_index_sel.value)
num_points = conductance_stat.count()['trace_index']
noise_stat = load_noise_data(direction_sel.value, file_index_sel.value)

axis_map = list(conductance_stat.columns)[1:]
x_axis_sel = Select(title="X Axis", options=axis_map, value="G_set")
y_axis_sel = Select(title="Y Axis", options=axis_map, value="G_stop")

bins_top, hist_top = utils.calc_hist_1d_single(conductance_stat[x_axis_sel.value][1:], xrange=(1e-7, 1),
                                               xbins_num=100, log_scale=True)
bins_right, hist_right = utils.calc_hist_1d_single(conductance_stat[y_axis_sel.value][1:], xrange=(1e-7, 1),
                                                   xbins_num=100, log_scale=True)
hist_top_sel = np.zeros_like(hist_top)
hist_right_sel = np.zeros_like(hist_right)

# Create Column Data Source that will be used by the plot
scatter_source = ColumnDataSource(data=dict(x=[], y=[], trace=[]))
hist_right_source = ColumnDataSource(data=dict(bins=[], hist_right=[]))
hist_top_source = ColumnDataSource(data=dict(bins=[], hist_top=[]))
hist_2d_G_source = ColumnDataSource(data=dict(image_plot=[], x=[], y=[], dw=[], dh=[]))
hist_2d_noise_source = ColumnDataSource(data=dict(image_plot=[], x=[], y=[], dw=[], dh=[]))

scatter_source.data = dict(
    x=conductance_stat[x_axis_sel.value],
    y=conductance_stat[y_axis_sel.value],
    trace=conductance_stat['trace_index']
)

hist_top_source.data = dict(
    bins=bins_top,
    hist_top=hist_top,
    hist_top_sel=hist_top_sel
)

hist_right_source.data = dict(
    bins=bins_right,
    hist_right=hist_right,
    hist_right_sel=hist_right_sel
)

scatter_plot = figure(width=500, height=500,
                      x_axis_type="log", y_axis_type="log",
                      tools="pan,box_zoom,lasso_select,reset,hover,xwheel_zoom,ywheel_zoom",
                      tooltips=[("Trace", "@trace"), (x_axis_sel.value, "@x"), (y_axis_sel.value, "@y")],
                      toolbar_location='left')

p1 = scatter_plot.circle("x", "y", source=scatter_source)

scatter_plot.line(x=np.logspace(start=-8, stop=1, num=50, base=10),
                  y=np.logspace(start=-8, stop=1, num=50, base=10),
                  color="#b8b7b4")

scatter_plot.xaxis.axis_label = x_axis_sel.value + " [G_0]"
scatter_plot.yaxis.axis_label = y_axis_sel.value + " [G_0]"
scatter_plot.xaxis.major_tick_in = 6
scatter_plot.xaxis.major_tick_out = 2
scatter_plot.xaxis.minor_tick_in = 4
scatter_plot.xaxis.minor_tick_out = 0
scatter_plot.yaxis.major_tick_in = 6
scatter_plot.yaxis.major_tick_out = 2
scatter_plot.yaxis.minor_tick_in = 4
scatter_plot.yaxis.minor_tick_out = 0

hist_top_plot = figure(width=500, height=250,
                       x_axis_type="log", x_axis_location="above", x_range=scatter_plot.x_range,
                       tools="pan,box_zoom,reset,hover,xwheel_zoom,ywheel_zoom", toolbar_location='left')
h11 = hist_top_plot.line("bins", "hist_top", source=hist_top_source)
h12 = hist_top_plot.line("bins", "hist_top_sel", source=hist_top_source, color='firebrick')

hist_top_plot.xaxis.axis_label = x_axis_sel.value + " [G_0]"
hist_top_plot.yaxis.axis_label = "Counts"
hist_top_plot.xaxis.major_tick_in = 6
hist_top_plot.xaxis.major_tick_out = 2
hist_top_plot.xaxis.minor_tick_in = 4
hist_top_plot.xaxis.minor_tick_out = 0
hist_top_plot.yaxis.major_tick_in = 6
hist_top_plot.yaxis.major_tick_out = 2
hist_top_plot.yaxis.minor_tick_in = 4
hist_top_plot.yaxis.minor_tick_out = 0

hist_right_plot = figure(width=250, height=500,
                         y_axis_type="log", y_axis_location="right", y_range=scatter_plot.y_range,
                         tools="pan,box_zoom,reset,hover,xwheel_zoom,ywheel_zoom", toolbar_location='right')
h21 = hist_right_plot.line("hist_right", "bins", source=hist_right_source)
h22 = hist_right_plot.line("hist_right_sel", "bins", source=hist_right_source, color='firebrick')

hist_right_plot.xaxis.axis_label = "Counts"
hist_right_plot.xaxis.major_tick_in = 6
hist_right_plot.xaxis.major_tick_out = 2
hist_right_plot.xaxis.minor_tick_in = 4
hist_right_plot.xaxis.minor_tick_out = 0
hist_right_plot.yaxis.axis_label = y_axis_sel.value + " [G_0]"
hist_right_plot.yaxis.major_tick_in = 6
hist_right_plot.yaxis.major_tick_out = 2
hist_right_plot.yaxis.minor_tick_in = 4
hist_right_plot.yaxis.minor_tick_out = 0

# 2d histogram

xbins_arr, ybins_arr, x_width, y_width, hist_2d_G = calc_2d_hist(conductance_stat[x_axis_sel.value][1:],
                                                               conductance_stat[y_axis_sel.value][1:],
                                                                 xrange=(1e-8, 10), log_scale_x=True,
                                                                 yrange=(1e-8, 10),
                                                                 num_bins=(10, 10))


hist_2d_G_range_slider = RangeSlider(start=min(hist_2d_G.flatten()), end=0.7 * max(hist_2d_G.flatten()),
                                     value=(1, 0.7 * max(hist_2d_G.flatten())), step=1,
                                     direction='rtl', orientation='vertical', title="Min..Max")

colors = list(map(convert_to_hex, test))
mapper = LinearColorMapper(palette=colors, low=hist_2d_G_range_slider.start, high=hist_2d_G_range_slider.end,
                           low_color=RGB(r=0, g=0, b=0, a=0))

mapper_grey = LinearColorMapper(palette=Greys256[::-1], low=hist_2d_G_range_slider.start, high=hist_2d_G_range_slider.end,
                                low_color=RGB(r=0, g=0, b=0, a=0))

hist_2d_G_source.data = dict(image_plot=[hist_2d_G],
                             x=[xbins_arr[0]],
                             y=[ybins_arr[0]],
                             dw=[xbins_arr[-1]],
                             dh=[ybins_arr[-1]])

hist_2d_G_range_slider.js_link("value", mapper, "low", attr_selector=0)
hist_2d_G_range_slider.js_link("value", mapper, "high", attr_selector=1)
hist_2d_G_range_slider.js_link("value", mapper_grey, "low", attr_selector=0)
hist_2d_G_range_slider.js_link("value", mapper_grey, "high", attr_selector=1)

hist_2d_G_plot = figure(width=500, height=500,
                        x_axis_type="log", y_axis_type="log", x_range=scatter_plot.x_range, y_range=scatter_plot.y_range,
                        tools="pan,box_zoom,reset,xwheel_zoom,ywheel_zoom")

hist_2d_G_plot.image(image='image_plot', x='x', y='y', dw='dw', dh='dh', level='image', color_mapper=mapper,
                     source=hist_2d_G_source)

hist_2d_G_plot.line(x=np.logspace(start=-8, stop=1, num=50, base=10),
                    y=np.logspace(start=-8, stop=1, num=50, base=10),
                    color="#b8b7b4")

def get_noise_type(noise_type, cols):
    if re.match(f'{noise_type}', cols) is not None:
        return True
    else:
        return False


def get_step(step, cols):
    if re.match(f'\w*_{step}', cols) is not None:
        return True
    else:
        return False


def get_avgs(cols):
    if re.match(f'avg', cols) is not None:
        return True
    else:
        return False


columns = np.array(list(noise_stat.columns))

avg_choices = columns[np.array(list(map(get_avgs, columns)))].tolist()
avg_sel = Select(title="Avg", options=avg_choices, value=avg_choices[0])
step_sel = utils.get_num_from_name(avg_sel.value)

noise_choices = columns[np.array(list(map(get_step,
                                          [str(step_sel)]*len(columns),
                                          columns)))][2:].tolist()

noise_sel = Select(title="Noise Type", options=noise_choices, value=noise_choices[0])

# 2d noise histogram
noise_x_range = (min(noise_stat[avg_sel.value][1:]) / float(fman(min(noise_stat[avg_sel.value][1:]))),
                 10 * max(noise_stat[avg_sel.value][1:]) / float(fman(max(noise_stat[avg_sel.value][1:]))))

noise_y_range = (min(noise_stat[noise_sel.value][1:]) / float(fman(min(noise_stat[noise_sel.value][1:]))),
                 10 * max(noise_stat[noise_sel.value][1:]) / float(fman(max(noise_stat[noise_sel.value][1:]))))

xbins_arr, ybins_arr, x_width, y_width, hist_2d_noise = calc_2d_hist(noise_stat[avg_sel.value][1:],
                                                                     noise_stat[noise_sel.value][1:],
                                                                     xrange=noise_x_range, log_scale_x=True,
                                                                     yrange=noise_y_range,
                                                                     num_bins=(10, 10))

hist_2d_noise_source.data = dict(image_plot=[hist_2d_noise],
                                 x=[xbins_arr[0]],
                                 y=[ybins_arr[0]],
                                 dw=[xbins_arr[-1]],
                                 dh=[ybins_arr[-1]])

hist_2d_noise_range_slider = RangeSlider(start=min(hist_2d_noise.flatten()), end=max(hist_2d_noise.flatten()),
                                         value=(1, max(hist_2d_noise.flatten())), step=1,
                                         direction='rtl', orientation='vertical', title="Min..Max")

mapper_noise = LinearColorMapper(palette=colors, low=hist_2d_noise_range_slider.start,
                                 high=hist_2d_noise_range_slider.end,
                                 low_color=RGB(r=0, g=0, b=0, a=0))

hist_2d_noise_range_slider.js_link("value", mapper_noise, "low", attr_selector=0)
hist_2d_noise_range_slider.js_link("value", mapper_noise, "high", attr_selector=1)

hist_2d_noise_plot = figure(width=500, height=500,
                            x_axis_type="log", y_axis_type="log",
                            tools="pan,box_zoom,reset,xwheel_zoom,ywheel_zoom", toolbar_location='left')

hist_2d_noise_plot.image(image='image_plot', x='x', y='y', dw='dw', dh='dh', level='image', color_mapper=mapper_noise,
                         source=hist_2d_noise_source)

hist_2d_noise_plot.line(x=np.logspace(start=-7, stop=0, num=50, base=10),
                        y=np.logspace(start=-13, stop=-6, num=50, base=10),
                        color="#b8b7b4")

hist_2d_noise_plot.line(x=np.logspace(start=-6, stop=-1, num=50, base=10),
                        y=np.logspace(start=-13, stop=-3, num=50, base=10),
                        color="#7287a1")

if re.match('avg_cond', noise_sel.value) is not None:
    hist_2d_noise_plot.xaxis.axis_label = "Avg Conductance [G_0]"
else:
    hist_2d_noise_plot.xaxis.axis_label = "Avg Current [A]"

if re.match('noise_power', noise_sel.value) is not None:
    hist_2d_noise_plot.yaxis.axis_label = "Noise Power [(G_0)^2]"
elif re.match('conductance', noise_sel.value) is not None:
    hist_2d_noise_plot.yaxis.axis_label = "Conductance Noise"
elif re.match('current', noise_sel.value) is not None:
    hist_2d_noise_plot.yaxis.axis_label = "Current Noise"

chosen_index = np.array([])

print(max(hist_2d_noise.flatten()))


def update_file_index():
    global conductance_stat, x_axis_sel, y_axis_sel, axis_map, direction_sel
    global noise_stat, columns, avg_choices, avg_sel, step_sel, noise_choices, noise_sel

    # load files with chosen index
    conductance_stat = load_conductance_data(direction_sel.value, file_index_sel.value)
    noise_stat = load_noise_data(direction_sel.value, file_index_sel.value)

    axis_map = list(conductance_stat.columns)[1:]
    x_axis_sel.options = axis_map
    x_axis_sel.value = axis_map[0]
    y_axis_sel.options = axis_map
    y_axis_sel.value = axis_map[1]

    columns = np.array(list(noise_stat.columns))

    avg_choices = columns[np.array(list(map(get_avgs, columns)))].tolist()
    avg_sel.options = avg_choices
    avg_sel.value = avg_choices[0]

    step_sel = utils.get_num_from_name(avg_sel.value)

    noise_choices = columns[np.array(list(map(get_step,
                                              [str(step_sel)] * len(columns),
                                              columns)))][2:].tolist()

    noise_sel.options = noise_choices
    noise_sel.value = noise_choices[0]


def update_axes():
    global chosen_index
    global hist_top, hist_top_sel
    global hist_right, hist_right_sel
    global hist_2d_G, mapper
    conductance_stat = load_conductance_data(direction_sel.value, file_index_sel.value)
    bins_top, hist_top = utils.calc_hist_1d_single(conductance_stat[x_axis_sel.value][1:], xrange=(1e-7, 1),
                                                   xbins_num=100, log_scale=True)
    bins_right, hist_right = utils.calc_hist_1d_single(conductance_stat[y_axis_sel.value][1:], xrange=(1e-7, 1),
                                                       xbins_num=100, log_scale=True)

    xbins_arr, ybins_arr, x_width, y_width, hist_2d_G = calc_2d_hist(conductance_stat[x_axis_sel.value][1:],
                                                                   conductance_stat[y_axis_sel.value][1:],
                                                                     xrange=(1e-8, 10), log_scale_x=True,
                                                                     yrange=(1e-8, 10),
                                                                     num_bins=(10, 10))

    hist_top_sel = np.zeros_like(hist_top)
    hist_right_sel = np.zeros_like(hist_right)

    if len(chosen_index) > 0:
        bins, hist_top_sel = utils.calc_hist_1d_single(conductance_stat[x_axis_sel.value][chosen_index],
                                                       xrange=(1e-7, 1), xbins_num=100, log_scale=True)
        bins, hist_right_sel = utils.calc_hist_1d_single(conductance_stat[y_axis_sel.value][chosen_index],
                                                         xrange=(1e-7, 1), xbins_num=100, log_scale=True)

        xbins_arr, ybins_arr, x_width, y_width, hist_2d_G = calc_2d_hist(
            conductance_stat[x_axis_sel.value][chosen_index],
            conductance_stat[y_axis_sel.value][chosen_index],
            xrange=(1e-8, 10), log_scale_x=True,
            yrange=(1e-8, 10),
            num_bins=(10, 10))

    hist_2d_G_range_slider.start = min(hist_2d_G.flatten())
    hist_2d_G_range_slider.end = 0.7 * max(hist_2d_G.flatten())
    hist_2d_G_range_slider.value = (1, 0.7 * max(hist_2d_G.flatten()))

    scatter_source.data = dict(
        x=conductance_stat[x_axis_sel.value],
        y=conductance_stat[y_axis_sel.value],
        trace=conductance_stat['trace_index']
    )

    hist_top_source.data = dict(
        bins=bins_top,
        hist_top=hist_top,
        hist_top_sel=hist_top_sel
    )

    hist_right_source.data = dict(
        bins=bins_right,
        hist_right=hist_right,
        hist_right_sel=hist_right_sel
    )

    hist_2d_G_source.data = dict(image_plot=[hist_2d_G],
                                 x=[xbins_arr[0]],
                                 y=[ybins_arr[0]],
                                 dw=[xbins_arr[-1]],
                                 dh=[ybins_arr[-1]])

    scatter_plot.xaxis.axis_label = x_axis_sel.value + " [G_0]"
    scatter_plot.yaxis.axis_label = y_axis_sel.value + " [G_0]"
    hist_top_plot.xaxis.axis_label = x_axis_sel.value + " [G_0]"
    hist_right_plot.yaxis.axis_label = y_axis_sel.value + " [G_0]"


def update_selection(attr, old, new):
    global chosen_index, num_points
    global hist_top, hist_top_sel
    global hist_right, hist_right_sel
    global hist_2d_G, mapper, mapper_grey
    bins_top, hist_top = utils.calc_hist_1d_single(conductance_stat[x_axis_sel.value][1:], xrange=(1e-7, 1),
                                                   xbins_num=100, log_scale=True)
    bins_right, hist_right = utils.calc_hist_1d_single(conductance_stat[y_axis_sel.value][1:], xrange=(1e-7, 1),
                                                       xbins_num=100, log_scale=True)

    xbins_arr, ybins_arr, x_width, y_width, hist_2d_G = calc_2d_hist(conductance_stat[x_axis_sel.value][1:],
                                                                   conductance_stat[y_axis_sel.value][1:],
                                                                     xrange=(1e-8, 10), log_scale_x=True,
                                                                     yrange=(1e-8, 10),
                                                                     num_bins=(10, 10))

    hist_2d_G_range_slider.start = min(hist_2d_G.flatten())
    hist_2d_G_range_slider.end = 0.7 * max(hist_2d_G.flatten())

    noise_x_range = (min(noise_stat[avg_sel.value][1:]) / float(fman(min(noise_stat[avg_sel.value][1:]))),
                     10 * max(noise_stat[avg_sel.value][1:]) / float(fman(max(noise_stat[avg_sel.value][1:]))))

    noise_y_range = (min(noise_stat[noise_sel.value][1:]) / float(fman(min(noise_stat[noise_sel.value][1:]))),
                     10 * max(noise_stat[noise_sel.value][1:]) / float(fman(max(noise_stat[noise_sel.value][1:]))))

    xbins_arr_n, ybins_arr_n, x_width_n, y_width_n, hist_2d_noise = calc_2d_hist(noise_stat[avg_sel.value][1:],
                                                                                 noise_stat[noise_sel.value][1:],
                                                                                 xrange=noise_x_range,
                                                                                 log_scale_x=True,
                                                                                 yrange=noise_y_range,
                                                                                 num_bins=(10, 10))

    hist_2d_noise_range_slider.start = min(hist_2d_noise.flatten())
    hist_2d_noise_range_slider.end = max(hist_2d_noise.flatten())
    print(max(hist_2d_noise.flatten()))
    inds = new
    chosen_index = inds
    print(f'Number of selected points: {len(chosen_index)}')
    if len(inds) == 0 or len(inds) == num_points:
        hist_top_sel = np.zeros_like(hist_top)
        hist_right_sel = np.zeros_like(hist_right)
    else:

        bins, hist_top_sel = utils.calc_hist_1d_single(conductance_stat[x_axis_sel.value][inds],
                                                       xrange=(1e-7, 1), xbins_num=100, log_scale=True)
        bins, hist_right_sel = utils.calc_hist_1d_single(conductance_stat[y_axis_sel.value][inds],
                                                         xrange=(1e-7, 1), xbins_num=100, log_scale=True)

        xbins_arr, ybins_arr, x_width, y_width, hist_2d_G = calc_2d_hist(conductance_stat[x_axis_sel.value][inds],
                                                                         conductance_stat[y_axis_sel.value][inds],
                                                                         xrange=(1e-8, 10), log_scale_x=True,
                                                                         yrange=(1e-8, 10),
                                                                         num_bins=(10, 10))

        noise_x_range = (min(noise_stat[avg_sel.value][inds]) / float(fman(min(noise_stat[avg_sel.value][inds]))),
                         10 * max(noise_stat[avg_sel.value][inds]) / float(fman(max(noise_stat[avg_sel.value][inds]))))

        noise_y_range = (min(noise_stat[noise_sel.value][inds]) / float(fman(min(noise_stat[noise_sel.value][inds]))),
                         10 * max(noise_stat[noise_sel.value][inds]) / float(fman(max(noise_stat[noise_sel.value][inds]))))

        xbins_arr_n, ybins_arr_n, x_width_n, y_width_n, hist_2d_noise = calc_2d_hist(noise_stat[avg_sel.value][inds],
                                                                                     noise_stat[noise_sel.value][inds],
                                                                                     xrange=noise_x_range,
                                                                                     log_scale_x=True,
                                                                                     yrange=noise_y_range,
                                                                                     num_bins=(10, 10))
    print('asd', max(hist_2d_noise.flatten()))
    hist_2d_noise_source.data = dict(image_plot=[hist_2d_noise],
                                     x=[xbins_arr_n[0]],
                                     y=[ybins_arr_n[0]],
                                     dw=[xbins_arr_n[-1]],
                                     dh=[ybins_arr_n[-1]])

    hist_2d_G_source.data = dict(image_plot=[hist_2d_G],
                                 x=[xbins_arr[0]],
                                 y=[ybins_arr[0]],
                                 dw=[xbins_arr[-1]],
                                 dh=[ybins_arr[-1]])

    hist_2d_G_plot.image(image='image_plot', x='x', y='y', dw='dw', dh='dh', level='image', color_mapper=mapper,
                         source=hist_2d_G_source)

    hist_top_source.data = dict(bins=bins_top,
                                hist_top=hist_top,
                                hist_top_sel=hist_top_sel)
    hist_right_source.data = dict(bins=bins_right,
                                  hist_right=hist_right,
                                  hist_right_sel=hist_right_sel)


def update_noise_type():
    global avg_sel
    global chosen_index

    step_sel = utils.get_num_from_name(avg_sel.value)

    noise_choices = columns[np.array(list(map(get_step,
                                              [str(step_sel)] * len(columns),
                                              columns)))][2:].tolist()

    noise_sel.options = noise_choices
    noise_sel.value = noise_choices[0]

    # 2d noise histogram
    noise_x_range = (min(noise_stat[avg_sel.value][1:]) / float(fman(min(noise_stat[avg_sel.value][1:]))),
                     10 * max(noise_stat[avg_sel.value][1:]) / float(fman(max(noise_stat[avg_sel.value][1:]))))

    noise_y_range = (min(noise_stat[noise_sel.value][1:]) / float(fman(min(noise_stat[noise_sel.value][1:]))),
                     10 * max(noise_stat[noise_sel.value][1:]) / float(fman(max(noise_stat[noise_sel.value][1:]))))

    xbins_arr, ybins_arr, x_width, y_width, hist_2d_noise = calc_2d_hist(noise_stat[avg_sel.value][1:],
                                                                         noise_stat[noise_sel.value][1:],
                                                                         xrange=noise_x_range, log_scale_x=True,
                                                                         yrange=noise_y_range,
                                                                         num_bins=(10, 10))

    hist_2d_noise_source.data = dict(image_plot=[hist_2d_noise],
                                     x=[xbins_arr[0]],
                                     y=[ybins_arr[0]],
                                     dw=[xbins_arr[-1]],
                                     dh=[ybins_arr[-1]])
    if re.match('avg_cond', avg_sel.value) is not None:
        hist_2d_noise_plot.xaxis.axis_label = "Avg Conductance [G_0]"
    else:
        hist_2d_noise_plot.xaxis.axis_label = "Avg Current [A]"

    if re.match('noise_power', noise_sel.value) is not None:
        hist_2d_noise_plot.yaxis.axis_label = "Noise Power [(G_0)^2]"
    elif re.match('conductance', noise_sel.value) is not None:
        hist_2d_noise_plot.yaxis.axis_label = "Conductance Noise"
    elif re.match('current', noise_sel.value) is not None:
        hist_2d_noise_plot.yaxis.axis_label = "Current Noise"


def update_noise_2d():
    global noise_stat, avg_sel, noise_sel, hist_2d_noise_source
    global chosen_index

    if len(chosen_index) > 0:
        noise_x_range = (min(noise_stat[avg_sel.value][chosen_index]) /
                         float(fman(min(noise_stat[avg_sel.value][chosen_index]))),
                         10 * max(noise_stat[avg_sel.value][chosen_index]) /
                         float(fman(max(noise_stat[avg_sel.value][chosen_index]))))
        noise_y_range = (min(noise_stat[noise_sel.value][chosen_index]) /
                         float(fman(min(noise_stat[noise_sel.value][chosen_index]))),
                         10 * max(noise_stat[noise_sel.value][chosen_index]) /
                         float(fman(max(noise_stat[noise_sel.value][chosen_index]))))

        xbins_arr, ybins_arr, x_width, y_width, hist_2d_noise = calc_2d_hist(noise_stat[avg_sel.value][chosen_index],
                                                                             noise_stat[noise_sel.value][chosen_index],
                                                                             xrange=noise_x_range, log_scale_x=True,
                                                                             yrange=noise_y_range,
                                                                             num_bins=(10, 10))
    else:
        noise_x_range = (min(noise_stat[avg_sel.value][1:]) / float(fman(min(noise_stat[avg_sel.value][1:]))),
                         10 * max(noise_stat[avg_sel.value][1:]) / float(fman(max(noise_stat[avg_sel.value][1:]))))
        noise_y_range = (min(noise_stat[noise_sel.value][1:]) / float(fman(min(noise_stat[noise_sel.value][1:]))),
                         10 * max(noise_stat[noise_sel.value][1:]) / float(fman(max(noise_stat[noise_sel.value][1:]))))

        xbins_arr, ybins_arr, x_width, y_width, hist_2d_noise = calc_2d_hist(noise_stat[avg_sel.value][1:],
                                                                             noise_stat[noise_sel.value][1:],
                                                                             xrange=noise_x_range, log_scale_x=True,
                                                                             yrange=noise_y_range,
                                                                             num_bins=(10, 10))

    hist_2d_noise_source.data = dict(image_plot=[hist_2d_noise],
                                     x=[xbins_arr[0]],
                                     y=[ybins_arr[0]],
                                     dw=[xbins_arr[-1]],
                                     dh=[ybins_arr[-1]])



scatter_plot.select(LassoSelectTool).select_every_mousemove = False

TOOLTIPS = []

controls = [direction_sel, x_axis_sel, y_axis_sel]

file_index_sel.on_change('value', (lambda attr, old, new: update_file_index()))
direction_sel.on_change('value', (lambda attr, old, new: update_file_index()))
avg_sel.on_change('value', (lambda attr, old, new: update_noise_type()))
noise_sel.on_change('value', (lambda attr, old, new: update_noise_2d()))
for control in controls:
    control.on_change('value', lambda attr, old, new: update_axes())

selection_glyph = Circle(fill_color='firebrick', fill_alpha=0.1, line_color=None)
nonselection_glyph = Circle(fill_color='#1f77b4', fill_alpha=0.1, line_color=None)
p1.selection_glyph = selection_glyph
p1.nonselection_glyph = nonselection_glyph
p1.data_source.selected.on_change('indices', update_selection)
h11.data_source.selected.on_change('indices', update_selection)
h12.data_source.selected.on_change('indices', update_selection)
h21.data_source.selected.on_change('indices', update_selection)
h22.data_source.selected.on_change('indices', update_selection)

controls_col = column(direction_sel, file_index_sel, x_axis_sel, y_axis_sel, width=200, height=500)

noise_controls_col = column(avg_sel, noise_sel, width=200, height=500)
grid = column(row(hist_top_plot, width=500),
              row(scatter_plot, hist_right_plot, hist_2d_G_plot, hist_2d_G_range_slider, width=1350),
              row(hist_2d_noise_plot, hist_2d_noise_range_slider, width=600), height=1250,
              sizing_mode="scale_both")
layout = row(column(controls_col, noise_controls_col), grid)

curdoc().add_root(layout)
curdoc().title = "Noise Analysis of Nanocontacts"
