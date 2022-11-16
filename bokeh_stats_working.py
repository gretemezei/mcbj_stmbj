import sys
import pandas as pd
import numpy as np
from pathlib import Path

import utils

from bokeh.io import output_notebook, show
# output_notebook()

from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, HoverTool, Span, Label,\
    LassoSelectTool, Select, LinearColorMapper, RangeSlider
from bokeh.models.glyphs import Circle
from bokeh.plotting import curdoc, figure, save
from bokeh.palettes import Blues

date = sys.argv[1]
home_folder = Path(f"D:/BJ_Data/{date}")

test = [(255, 255, 255), (255, 235, 235), (255, 215, 215), (255, 196, 196), (245, 179, 174), (255, 158, 158),
        (255, 124, 124), (255, 90, 90), (238, 80, 78), (244, 117, 75), (255, 160, 69), (255, 189, 87), (247, 215, 104),
        (240, 236, 121), (223, 245, 141),  (205, 255, 162), (172, 245, 168), (138, 236, 174), (124, 235, 200),
        (106, 235, 225), (97, 225, 240), (68, 202, 255), (50, 190, 255), (25, 175, 255), (13, 129, 248), (26, 102, 240),
        (0, 40, 224), (0, 25, 212), (0, 10, 200), (20, 5, 175), (40, 0, 150), (10, 0, 121)]


def convert_to_hex(arr):
    return '#%02x%02x%02x' % arr


def load_conductance_data(direction, file_index):
    return pd.read_csv(home_folder.joinpath(f"results/old/conductance_stats_{direction}_{file_index}.csv"))


def load_noise_data(direction, file_index):
    return pd.read_csv(home_folder.joinpath(f"results/old/noise_stats_{direction}_{file_index}.csv"))


def calc_2d_hist(x_values, y_values, xrange, log_scale_x = False,
                 yrange = (1e-7, 10), num_bins=(100, 100)):
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

    x_width=np.diff(xedges)
    y_width=np.diff(yedges)

    # xbins_arr = xedges[:-1]
    # ybins_arr = yedges[:-1]

    return xbins_arr, ybins_arr, x_width, y_width, h.T


file_names = np.array(list(map(utils.get_name_from_path, home_folder.joinpath('results/old').glob(r'*.csv'))))
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
hist_2d_source = ColumnDataSource(data=dict(x=[], y=[], hist_2d=[]))

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
                      tools="pan,box_zoom,lasso_select,reset,hover",
                      tooltips=[("Trace", "@trace"), (x_axis_sel.value, "@x"), (y_axis_sel.value, "@y")],
                      toolbar_location='left')

p1 = scatter_plot.circle("x", "y", source=scatter_source)

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
                       tools="pan,box_zoom,reset,hover", toolbar_location='left')
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
                         tools="pan,box_zoom,reset,hover", toolbar_location='right')
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

hist_2d_range_slider = RangeSlider(start=min(hist_2d_G.flatten()), end=0.4 * max(hist_2d_G.flatten()),
                                   value=(1, 0.4*max(hist_2d_G.flatten())), step=1,
                                   direction='rtl', orientation='vertical', title="Stuff")

colors = list(map(convert_to_hex, test))
mapper = LinearColorMapper(palette=colors, low=hist_2d_range_slider.start, high=hist_2d_range_slider.end,
                           low_color='white')

hist_2d_range_slider.js_link("value", mapper, "low", attr_selector=0)
hist_2d_range_slider.js_link("value", mapper, "high", attr_selector=1)

hist_2d_plot = figure(width=500, height=500,
                      x_axis_type="log", y_axis_type="log", x_range=scatter_plot.x_range, y_range=scatter_plot.y_range,
                      tools="pan,box_zoom,reset,hover")

hist_2d_plot.image(image=[hist_2d_G], x=xbins_arr[0], y=ybins_arr[0], dw=xbins_arr[-1], dh=ybins_arr[-1], level='image',
                   color_mapper=mapper)

chosen_index = np.array([])


def update_file_index():
    global conductance_stat
    global x_axis_sel
    global y_axis_sel
    global axis_map

    conductance_stat = load_conductance_data(direction_sel.value, file_index_sel.value)

    axis_map = list(conductance_stat.columns)[1:]
    x_axis_sel.options = axis_map
    x_axis_sel.value = axis_map[0]
    y_axis_sel.options = axis_map
    y_axis_sel.value = axis_map[1]


def update():
    global hist_top, hist_top_sel
    global hist_right, hist_right_sel
    global hist_2d_G, mapper
    conductance_stat = load_conductance_data(direction_sel.value, file_index_sel.value)
    bins_top, hist_top = utils.calc_hist_1d_single(conductance_stat[x_axis_sel.value][1:], xrange=(1e-7, 1),
                                                   xbins_num=100, log_scale=True)
    bins_right, hist_right = utils.calc_hist_1d_single(conductance_stat[y_axis_sel.value][1:], xrange=(1e-7, 1),
                                                       xbins_num=100, log_scale=True)

    hist_top_sel = np.zeros_like(hist_top)
    hist_right_sel = np.zeros_like(hist_right)

    if len(chosen_index) > 0:
        bins, hist_top_sel = utils.calc_hist_1d_single(conductance_stat[x_axis_sel.value][chosen_index],
                                                       xrange=(1e-7, 1), xbins_num=100, log_scale=True)
        bins, hist_right_sel = utils.calc_hist_1d_single(conductance_stat[y_axis_sel.value][chosen_index],
                                                         xrange=(1e-7, 1), xbins_num=100, log_scale=True)

    xbins_arr, ybins_arr, x_width, y_width, hist_2d = calc_2d_hist(conductance_stat[x_axis_sel.value][1:],
                                                                   conductance_stat[y_axis_sel.value][1:],
                                                                   xrange=(1e-8, 10), log_scale_x=True,
                                                                   yrange=(1e-8, 10),
                                                                   num_bins=(10, 10))
    hist_2d_plot.image(image=[hist_2d], x=xbins_arr[0], y=ybins_arr[0], dw=xbins_arr[-1], dh=ybins_arr[-1],
                       level='image', color_mapper=mapper)

    hist_2d_range_slider.start = min(hist_2d.flatten())
    hist_2d_range_slider.end = 0.4*max(hist_2d.flatten())
    # hist_2d_range_slider.value = (1, 0.25*max(hist_2d.flatten()))

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

    scatter_plot.xaxis.axis_label = x_axis_sel.value + " [G_0]"
    scatter_plot.yaxis.axis_label = y_axis_sel.value + " [G_0]"
    hist_top_plot.xaxis.axis_label = x_axis_sel.value + " [G_0]"
    hist_right_plot.yaxis.axis_label = y_axis_sel.value + " [G_0]"


def update_selection(attr, old, new):
    global chosen_index
    global hist_top, hist_top_sel
    global hist_right, hist_right_sel
    bins_top, hist_top = utils.calc_hist_1d_single(conductance_stat[x_axis_sel.value][1:], xrange=(1e-7, 1),
                                                   xbins_num=100, log_scale=True)
    bins_right, hist_right = utils.calc_hist_1d_single(conductance_stat[y_axis_sel.value][1:], xrange=(1e-7, 1),
                                                       xbins_num=100, log_scale=True)

    inds = new
    chosen_index = inds
    if len(inds) == 0 or len(inds) == num_points:
        hist_top_sel = np.zeros_like(hist_top)
        hist_right_sel = np.zeros_like(hist_right)
    else:

        bins, hist_top_sel = utils.calc_hist_1d_single(conductance_stat[x_axis_sel.value][inds],
                                                       xrange=(1e-7, 1), xbins_num=100, log_scale=True)
        bins, hist_right_sel = utils.calc_hist_1d_single(conductance_stat[y_axis_sel.value][inds],
                                                         xrange=(1e-7, 1), xbins_num=100, log_scale=True)

    hist_top_source.data = dict(bins=bins_top,
                                hist_top=hist_top,
                                hist_top_sel=hist_top_sel)
    hist_right_source.data = dict(bins=bins_right,
                                  hist_right=hist_right,
                                  hist_right_sel=hist_right_sel)


scatter_plot.select(LassoSelectTool).select_every_mousemove = False

TOOLTIPS = []

controls = [direction_sel, x_axis_sel, y_axis_sel]

file_index_sel.on_change('value', (lambda attr, old, new: update_file_index()))
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

selection_glyph = Circle(fill_color='firebrick', fill_alpha=0.1, line_color=None)
nonselection_glyph = Circle(fill_color='#1f77b4', fill_alpha=0.1, line_color=None)
p1.selection_glyph = selection_glyph
p1.nonselection_glyph = nonselection_glyph
p1.data_source.selected.on_change('indices', update_selection)
h11.data_source.selected.on_change('indices', update_selection)
h12.data_source.selected.on_change('indices', update_selection)
h21.data_source.selected.on_change('indices', update_selection)
h22.data_source.selected.on_change('indices', update_selection)

controls_col = column(direction_sel, file_index_sel, x_axis_sel, y_axis_sel, width=100)
grid = column(row(hist_top_plot, width=500),
              row(scatter_plot, hist_right_plot, hist_2d_plot, hist_2d_range_slider, width=1350), height=750, sizing_mode="scale_both")
layout = row(controls_col, grid)

curdoc().add_root(layout)
curdoc().title = "Test"