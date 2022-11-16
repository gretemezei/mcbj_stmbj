import pandas as pd
import numpy as np
from pathlib import Path

import utils

from bokeh.io import output_notebook, show
# output_notebook()

from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, CDSView, GroupFilter, Paragraph, HoverTool, Span, Label,\
    LassoSelectTool
from bokeh.models.glyphs import Circle
from bokeh.plotting import curdoc, figure, save

date = "21_12_08"
sample_rate = 50_000
home_folder = Path(f"D:/BJ_Data/{date}")

# read data from csv
conductance_stat_pull = pd.read_csv(home_folder.joinpath("results/conductance_stat_pull_1.csv"))
conductance_stat_push = pd.read_csv(home_folder.joinpath("results/conductance_stat_push_1.csv"))
noise_stat_pull = pd.read_csv(home_folder.joinpath("results/noise_stats_pull_1.csv"))
noise_stat_push = pd.read_csv(home_folder.joinpath("results/noise_stats_push_1.csv"))


num_points = conductance_stat_pull.count()['trace_index']

cds_pull = ColumnDataSource(conductance_stat_pull)
cds_push = ColumnDataSource(conductance_stat_push)

bins_pull, hist_G_set_pull_tot = utils.calc_hist_1d_single(conductance_stat_pull['G_set'], xrange=(1e-7, 1),
                                                           xbins_num=100, log_scale=True)
bins_pull, hist_G_stop_pull_tot = utils.calc_hist_1d_single(conductance_stat_pull['G_stop'], xrange=(1e-7, 1),
                                                            xbins_num=100, log_scale=True)
bins_pull, hist_G_hold_pull_tot = utils.calc_hist_1d_single(conductance_stat_pull['G_hold'], xrange=(1e-7, 1),
                                                            xbins_num=100, log_scale=True)
bins_pull, hist_G_avg_pull_tot = utils.calc_hist_1d_single(conductance_stat_pull['G_avg'], xrange=(1e-7, 1),
                                                           xbins_num=100, log_scale=True)

cds_hist = ColumnDataSource(data=dict(bins = bins_pull,
                                      hist_G_set=hist_G_set_pull_tot,
                                      hist_G_stop=hist_G_stop_pull_tot,
                                      hist_G_hold=hist_G_hold_pull_tot,
                                      hist_G_avg=hist_G_avg_pull_tot))

hist_G_set_pull_sel = np.zeros_like(hist_G_set_pull_tot)
hist_G_stop_pull_sel = np.zeros_like(hist_G_stop_pull_tot)
hist_G_hold_pull_sel = np.zeros_like(hist_G_hold_pull_tot)
hist_G_avg_pull_sel = np.zeros_like(hist_G_avg_pull_tot)

cds_hist_sel = ColumnDataSource(data=dict(bins = bins_pull,
                                          hist_G_set=hist_G_set_pull_sel,
                                          hist_G_stop=hist_G_stop_pull_sel,
                                          hist_G_hold=hist_G_hold_pull_sel,
                                          hist_G_avg=hist_G_avg_pull_sel))

my_size = 3
figsize = 150

# row 1
s01 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", x_axis_location="above",
             tools="pan,box_zoom,reset,hover", tooltips=[("G", "@bins"), ("count", "@hist_G_set")])
r01_tot = s01.line(x="bins", y="hist_G_set", source=cds_hist, color='#1f77b4', alpha=0.3)
r01_sel = s01.line(x="bins", y="hist_G_set", source=cds_hist_sel, color='firebrick')

s02 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_type="log", x_axis_location="above",
             tools="pan,box_zoom,lasso_select,reset,hover",
             tooltips=[("Trace", "@trace_index"), ("G_set", "@G_set"), ("G_stop", "@G_stop")])
r02 = s02.circle('G_stop', 'G_set', source=cds_pull, size=my_size, color='#1f77b4', alpha=0.5)

s03 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_type="log", x_axis_location="above",
             tools="pan,box_zoom,lasso_select,reset,hover",
             tooltips=[("Trace", "@trace_index"), ("G_set", "@G_set"), ("G_hold", "@G_hold")],
             y_range=s02.y_range)
r03 = s03.circle('G_hold', 'G_set', source=cds_pull, size=my_size, color='#1f77b4', alpha=0.5)

s04 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_type="log", x_axis_location="above", y_axis_location="right",
             tools="pan,box_zoom,lasso_select,reset,hover",
             tooltips=[("Trace", "@trace_index"), ("G_set", "@G_set"), ("G_avg", "@G_avg")],
             y_range=s02.y_range)
r04 = s04.circle('G_avg', 'G_set', source=cds_pull, size=my_size, color='#1f77b4', alpha=0.5)

# row 2
s05 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_type="log",
             tools="pan,box_zoom,lasso_select,reset,hover",
             tooltips=[("Trace", "@trace_index"), ("G_stop", "@G_stop"), ("G_set", "@G_set")],
             x_range=s01.x_range)
r05 = s05.circle('G_set', 'G_stop', source=cds_pull, size=my_size, color='#1f77b4', alpha=0.5)

s06 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log",
             tools="pan,box_zoom,reset,hover", tooltips=[("G", "@bins"), ("count", "@hist_G_stop")],
             x_range=s02.x_range)
r06_tot = s06.line(x="bins", y="hist_G_stop", source=cds_hist, color='#1f77b4', alpha=0.3)
r06_sel = s06.line(x="bins", y="hist_G_stop", source=cds_hist_sel, color='firebrick')

s07 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_type="log",
             tools="pan,box_zoom,lasso_select,reset,hover",
             tooltips=[("Trace", "@trace_index"), ("G_hold", "@G_hold"), ("G_stop", "@G_stop")],
             x_range=s03.x_range,
             y_range=s05.y_range)
r07 = s07.circle('G_hold', 'G_stop', source=cds_pull, size=my_size, color='#1f77b4', alpha=0.5)

s08 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_type="log", y_axis_location="right",
             tools="pan,box_zoom,lasso_select,reset,hover",
             tooltips=[("Trace", "@trace_index"), ("G_avg", "@G_avg"), ("G_stop", "@G_stop")],
             x_range=s04.x_range,
             y_range=s05.y_range)
r08 = s08.circle('G_avg', 'G_stop', source=cds_pull, size=my_size, color='#1f77b4', alpha=0.5)

# row 3
s09 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_type="log",
             tools="pan,box_zoom,lasso_select,reset,hover",
             tooltips=[("Trace", "@trace_index"), ("G_set", "@G_set"), ("G_hold", "@G_hold")],
             x_range=s01.x_range)
r09 = s09.circle('G_set', 'G_hold', source=cds_pull, size=my_size, color='#1f77b4', alpha=0.5)

s10 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_type="log",
             x_range=s02.x_range,
             y_range=s09.y_range,
             tools="pan,box_zoom,lasso_select,reset,hover",
             tooltips=[("Trace", "@trace_index"), ("G_stop", "@G_stop"), ("G_hold", "@G_hold")])
r10 = s10.circle('G_stop', 'G_hold', source=cds_pull, size=my_size, color='#1f77b4', alpha=0.5)

s11 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log",
             tools="pan,box_zoom,reset,hover", tooltips=[("G", "@bins"), ("count", "@hist_G_hold")],
             x_range=s03.x_range)
r11_tot = s11.line(x="bins", y="hist_G_hold", source=cds_hist, color='#1f77b4', alpha=0.3)
r11_sel = s11.line(x="bins", y="hist_G_hold", source=cds_hist_sel, color='firebrick')

s12 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_type="log", y_axis_location="right",
             tools="pan,box_zoom,lasso_select,reset,hover",
             tooltips=[("Trace", "@trace_index"), ("G_avg", "@G_avg"), ("G_hold", "@G_hold")],
             x_range=s04.x_range,
             y_range=s09.y_range)
r12 = s12.circle('G_avg', 'G_hold', source=cds_pull, size=my_size, color='#1f77b4', alpha=0.5)

# row 4
s13 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_type="log",
             tools="pan,box_zoom,lasso_select,reset,hover",
             x_range=s01.x_range)
r13 = s13.circle('G_set', 'G_avg', source=cds_pull, size=my_size, color='#1f77b4', alpha=0.5)

s14 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_type="log",
             x_range=s02.x_range,
             y_range=s13.y_range,
             tools="pan,box_zoom,lasso_select,reset,hover",
             tooltips=[("Trace", "@trace_index"), ("G_stop", "@G_stop"), ("G_avg", "@G_avg")])
r14 = s14.circle('G_stop', 'G_avg', source=cds_pull, size=my_size, color='#1f77b4', alpha=0.5)

s15 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_type="log",
             tools="pan,box_zoom,lasso_select,reset,hover",
             tooltips=[("Trace", "@trace_index"), ("G_hold", "@G_hold"), ("G_avg", "@G_avg")],
             x_range=s03.x_range,
             y_range=s13.y_range)
r15 = s15.circle('G_hold', 'G_avg', source=cds_pull, size=my_size, color='#1f77b4', alpha=0.5)

s16 = figure(width=figsize, height=figsize, background_fill_color="#fafafa",
             x_axis_type="log", y_axis_location="right",
             tools="pan,box_zoom,reset,hover", tooltips=[("G", "@bins"), ("count", "@hist_G_avg")],
             x_range=s04.x_range)
r16_tot = s16.line(x="bins", y="hist_G_avg", source=cds_hist, color='#1f77b4', alpha=0.3)
r16_sel = s16.line(x="bins", y="hist_G_avg", source=cds_hist_sel, color='firebrick')

grid = gridplot([[s01, s02, s03, s04],
                 [s05, s06, s07, s08],
                 [s09, s10, s11, s12],
                 [s13, s14, s15, s16]], sizing_mode="scale_both")

s02.select(LassoSelectTool).select_every_mousemove = False
s03.select(LassoSelectTool).select_every_mousemove = False
s04.select(LassoSelectTool).select_every_mousemove = False
s05.select(LassoSelectTool).select_every_mousemove = False
s07.select(LassoSelectTool).select_every_mousemove = False
s08.select(LassoSelectTool).select_every_mousemove = False
s09.select(LassoSelectTool).select_every_mousemove = False
s10.select(LassoSelectTool).select_every_mousemove = False
s12.select(LassoSelectTool).select_every_mousemove = False
s13.select(LassoSelectTool).select_every_mousemove = False
s14.select(LassoSelectTool).select_every_mousemove = False
s15.select(LassoSelectTool).select_every_mousemove = False

selection_glyph = Circle(fill_color='firebrick', line_color=None)
nonselection_glyph = Circle(fill_color='#1f77b4', fill_alpha=0.3, line_color=None)

s01.xaxis.axis_label = "G_set [G_0]"
s01.yaxis.axis_label = "Counts"
s02.xaxis.axis_label = "G_stop [G_0]"
s02.yaxis.axis_label = "G_set [G_0]"
s03.xaxis.axis_label = "G_hold [G_0]"
s03.yaxis.axis_label = "G_set [G_0]"
s04.xaxis.axis_label = "G_avg [G_0]"
s04.yaxis.axis_label = "G_set [G_0]"

s05.xaxis.axis_label = "G_set [G_0]"
s05.yaxis.axis_label = "G_stop [G_0]"
s06.xaxis.axis_label = "G_stop [G_0]"
s06.yaxis.axis_label = "Counts"
s07.xaxis.axis_label = "G_hold [G_0]"
s07.yaxis.axis_label = "G_stop [G_0]"
s08.xaxis.axis_label = "G_avg [G_0]"
s08.yaxis.axis_label = "G_stop [G_0]"

s09.xaxis.axis_label = "G_set [G_0]"
s09.yaxis.axis_label = "G_hold [G_0]"
s10.xaxis.axis_label = "G_stop [G_0]"
s10.yaxis.axis_label = "G_hold [G_0]"
s11.xaxis.axis_label = "G_hold [G_0]"
s11.yaxis.axis_label = "Counts"
s12.xaxis.axis_label = "G_avg [G_0]"
s12.yaxis.axis_label = "G_hold [G_0]"

s13.xaxis.axis_label = "G_set [G_0]"
s13.yaxis.axis_label = "G_avg [G_0]"
s14.xaxis.axis_label = "G_stop [G_0]"
s14.yaxis.axis_label = "G_avg [G_0]"
s15.xaxis.axis_label = "G_hold [G_0]"
s15.yaxis.axis_label = "G_avg [G_0]"
s16.xaxis.axis_label = "G_avg [G_0]"
s16.yaxis.axis_label = "Counts"

r02.selection_glyph = selection_glyph
r02.nonselection_glyph = nonselection_glyph
r03.selection_glyph = selection_glyph
r03.nonselection_glyph = nonselection_glyph
r04.selection_glyph = selection_glyph
r04.nonselection_glyph = nonselection_glyph
r05.selection_glyph = selection_glyph
r05.nonselection_glyph = nonselection_glyph
r07.selection_glyph = selection_glyph
r07.nonselection_glyph = nonselection_glyph
r08.selection_glyph = selection_glyph
r08.nonselection_glyph = nonselection_glyph
r09.selection_glyph = selection_glyph
r09.nonselection_glyph = nonselection_glyph
r10.selection_glyph = selection_glyph
r10.nonselection_glyph = nonselection_glyph
r12.selection_glyph = selection_glyph
r12.nonselection_glyph = nonselection_glyph
r13.selection_glyph = selection_glyph
r13.nonselection_glyph = nonselection_glyph
r14.selection_glyph = selection_glyph
r14.nonselection_glyph = nonselection_glyph
r15.selection_glyph = selection_glyph
r15.nonselection_glyph = nonselection_glyph


def update(attr, old, new):
    inds = new
    if len(inds) == 0 or len(inds) == num_points:
        hist_G_set_pull_sel = np.zeros_like(hist_G_set_pull_tot)
        hist_G_stop_pull_sel = np.zeros_like(hist_G_stop_pull_tot)
        hist_G_hold_pull_sel = np.zeros_like(hist_G_hold_pull_tot)
        hist_G_avg_pull_sel = np.zeros_like(hist_G_avg_pull_tot)
    else:
        neg_inds = np.ones(num_points, dtype=np.bool)
        neg_inds[inds] = False

        bins_pull, hist_G_set_pull_sel = utils.calc_hist_1d_single(conductance_stat_pull['G_set'][inds],
                                                                   xrange=(1e-7, 1), xbins_num=100, log_scale=True)
        bins_pull, hist_G_stop_pull_sel = utils.calc_hist_1d_single(conductance_stat_pull['G_stop'][inds],
                                                                    xrange=(1e-7, 1), xbins_num=100, log_scale=True)
        bins_pull, hist_G_hold_pull_sel = utils.calc_hist_1d_single(conductance_stat_pull['G_hold'][inds],
                                                                    xrange=(1e-7, 1), xbins_num=100, log_scale=True)
        bins_pull, hist_G_avg_pull_sel = utils.calc_hist_1d_single(conductance_stat_pull['G_avg'][inds],
                                                                   xrange=(1e-7, 1), xbins_num=100, log_scale=True)

    cds_hist_sel.data = dict(bins=bins_pull,
                             hist_G_set=hist_G_set_pull_sel,
                             hist_G_stop=hist_G_stop_pull_sel,
                             hist_G_hold=hist_G_hold_pull_sel,
                             hist_G_avg=hist_G_avg_pull_sel)

r01_sel.data_source.selected.on_change('indices', update)
r02.data_source.selected.on_change('indices', update)
r03.data_source.selected.on_change('indices', update)
r04.data_source.selected.on_change('indices', update)
r05.data_source.selected.on_change('indices', update)
r06_sel.data_source.selected.on_change('indices', update)
r07.data_source.selected.on_change('indices', update)
r08.data_source.selected.on_change('indices', update)
r09.data_source.selected.on_change('indices', update)
r10.data_source.selected.on_change('indices', update)
r11_sel.data_source.selected.on_change('indices', update)
r12.data_source.selected.on_change('indices', update)
r13.data_source.selected.on_change('indices', update)
r14.data_source.selected.on_change('indices', update)
r15.data_source.selected.on_change('indices', update)
r16_sel.data_source.selected.on_change('indices', update)

# show(grid)
curdoc().add_root(grid)
curdoc().title = "Test"