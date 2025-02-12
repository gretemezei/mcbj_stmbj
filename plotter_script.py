from mcbj import *
import plots
import filter_traces
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# import pandas as pd

matplotlib.use('Agg')

date = "23_09_20"
home_folder = Path(f'//DATACENTER/BreakJunction_group/BJ_Data/{date}')

with open(home_folder.joinpath('results/did_not_break_pull.npy'), 'rb') as f:
    iv_good_pull = np.load(f)

with open(home_folder.joinpath('results/did_not_break_push.npy'), 'rb') as f:
    iv_good_push = np.load(f)


warnings.filterwarnings("ignore")

# for which_trace in tqdm(iv_good_pull, desc='Saving pull plots'):
#     trace_pair = TracePair(which_trace, load_from=home_folder)
#
#     hold_trace = HoldTrace(which_trace,
#                            load_from=home_folder, bias_offset=0,
#                            r_serial_ohm=trace_pair.R_serial, min_step_len=6_000, min_height=1, iv=0)
#
#     hold_trace.analyse_hold_trace(num_of_fft=4, subtract_bg=False)
#
#     hold_trace.save_iv_for_laci(home_folder=home_folder, direction='pull')
#
#     plots.plot_iv_with_details(hold_trace=hold_trace, trace_pair=trace_pair,
#                                direction='pull', smoothing=20, cmap='inferno')
#
#     plt.savefig(home_folder.joinpath(f'results/figs/pull_{which_trace}.png'), bbox_inches='tight')
#
#     plt.clf()
#     plt.close(fig='all')
#
# for which_trace in tqdm(iv_good_push, desc='Saving push plots'):
#     trace_pair = TracePair(which_trace, load_from=home_folder)
#
#     hold_trace = HoldTrace(which_trace,
#                            load_from=home_folder, bias_offset=0,
#                            r_serial_ohm=trace_pair.R_serial, min_step_len=6_000, min_height=1, iv=0)
#
#     hold_trace.analyse_hold_trace(num_of_fft=4, subtract_bg=False)
#
#     hold_trace.save_iv_for_laci(home_folder=home_folder, direction='push')
#
#     plots.plot_iv_with_details(hold_trace=hold_trace, trace_pair=trace_pair,
#                                direction='push', smoothing=20, cmap='inferno')
#
#     plt.savefig(home_folder.joinpath(f'results/figs/push_{which_trace}.png'), bbox_inches='tight')
#
#     plt.clf()
#     plt.close(fig='all')

def plot_half_ivs(trace_num,
                  direction,
                  main_colors: Tuple[str, str] = ('cornflowerblue', 'indianred'),
                  accent_colors: Tuple[str, str] = ('royalblue', 'firebrick'),
                  vline_color: str = 'grey',
                  color_list: Optional[Union[List, np.ndarray]] = None,
                  smoothing: int = 1,
                  iv_num_xticks: int = 5,
                  which_psds: Optional[List[int]] = None,
                  plot_mean_current=False,
                  fig_size: Optional[Tuple[float, float]] = utils.cm2inch(16, 10)):
    hold_trace = HoldTrace(trace_num,
                           load_from=home_folder, bias_offset=0,
                           r_serial_ohm=100_100, min_step_len=4_000, min_height=1)

    hold_trace.analyse_hold_trace(num_of_fft=1, subtract_bg=False)

    trace_pair = TracePair(trace_num, load_from=home_folder)

    if direction == 'pull':
        iv_current_1 = hold_trace.hold_current_pull[hold_trace.bias_steps_ranges_pull[0][1]:
                                                    hold_trace.bias_steps_ranges_pull[1][0]]

        iv_current_2 = hold_trace.hold_current_pull[hold_trace.bias_steps_ranges_pull[-1][1]:]

        iv_bias_1 = hold_trace.hold_bias_pull[hold_trace.bias_steps_ranges_pull[0][1]:
                                              hold_trace.bias_steps_ranges_pull[1][0]]

        iv_bias_2 = hold_trace.hold_bias_pull[hold_trace.bias_steps_ranges_pull[-1][1]:]

        avg_current_on_step = hold_trace.avg_current_on_step_pull
        hold_trace_areas = hold_trace.areas_pull
        hold_trace_current_noise = hold_trace.current_noise_pull

        if color_list is None:
            cols = utils.blues
        else:
            cols = color_list
        pull = True
        if which_psds is None:
            psd_intervals = hold_trace.psd_interval_ranges_pull
        else:
            psd_intervals = hold_trace.psd_interval_ranges_pull[which_psds]
        col_index = 0
    elif direction == 'push':
        iv_current_1 = hold_trace.hold_current_push[hold_trace.bias_steps_ranges_push[0][1]:
                                                    hold_trace.bias_steps_ranges_push[1][0]]

        iv_current_2 = hold_trace.hold_current_push[hold_trace.bias_steps_ranges_push[-1][1]:]

        iv_bias_1 = hold_trace.hold_bias_push[hold_trace.bias_steps_ranges_push[0][1]:
                                              hold_trace.bias_steps_ranges_push[1][0]]

        iv_bias_2 = hold_trace.hold_bias_push[hold_trace.bias_steps_ranges_push[-1][1]:]

        avg_current_on_step = hold_trace.avg_current_on_step_push
        hold_trace_areas = hold_trace.areas_push
        hold_trace_current_noise = hold_trace.current_noise_push

        if color_list is None:
            cols = utils.reds
        else:
            cols = color_list
        pull = False
        if which_psds is None:
            psd_intervals = hold_trace.psd_interval_ranges_push
        else:
            psd_intervals = hold_trace.psd_interval_ranges_push[which_psds]
        col_index = 1
    else:
        raise ValueError('Invalid value for parameter `direction.`')

    fig = plt.figure(figsize=fig_size, dpi=600)  # figsize: (width, height) in inches

    gs_total = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=(1, 1),
                                 figure=fig, left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.15)

    gs_top = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, width_ratios=(1, 2),
                                              subplot_spec=gs_total[0],
                                              wspace=0.25, hspace=0)

    gs_bottom = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, width_ratios=(1, 1),
                                                 subplot_spec=gs_total[1],
                                                 wspace=0.1, hspace=0)

    ax_trace = fig.add_subplot(gs_top[0])
    ax_hold = fig.add_subplot(gs_top[1])

    ax_iv = fig.add_subplot(gs_bottom[0])
    ax_psd = fig.add_subplot(gs_bottom[1])

    ax_noise_1 = ax_iv.inset_axes(bounds=(0.1, 0.05, 0.35, 0.35))
    ax_noise_2 = ax_iv.inset_axes(bounds=(0.5, 0.05, 0.35, 0.35))

    ax_trace.xaxis.set_label_position('top')
    ax_trace.xaxis.tick_top()
    ax_trace.xaxis.set_ticks_position('both')
    ax_trace.yaxis.set_ticks_position('both')

    ax_hold.xaxis.set_ticks_position('both')
    ax_hold.xaxis.set_label_position('top')
    ax_hold.xaxis.tick_top()

    ax_iv.xaxis.set_ticks_position('both')
    ax_iv.yaxis.set_ticks_position('both')

    ax_psd.xaxis.set_ticks_position('both')
    ax_psd.yaxis.tick_right()
    ax_psd.yaxis.set_ticks_position('both')
    ax_psd.yaxis.set_label_position('right')

    ax_trace = trace_pair.plot_trace_pair(ax=ax_trace, xlim=None,
                                          main_colors=main_colors,
                                          accent_colors=accent_colors,
                                          smoothing=smoothing, plot_trigger=True)

    popt, perr = trace_pair.fit_tunnel()

    ax_trace.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))
    ax_trace.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))

    # IV

    ax_iv.plot(utils.moving_average(iv_bias_1, smoothing),
               utils.moving_average(iv_current_1, smoothing),
               c=main_colors[col_index], lw=0.4)

    ax_iv.plot(utils.moving_average(iv_bias_2, smoothing),
               utils.moving_average(iv_current_2, smoothing),
               c=accent_colors[col_index], lw=0.4)

    ax_iv.set_xlabel('Bias [V]')
    ax_iv.set_ylabel('Current [A]')

    max_curr = max((max(abs(iv_current_1)),
                    max(abs(iv_current_2))))

    ax_iv.set_ylim(-1 * max_curr, max_curr)
    ax_iv.set_yticks(np.linspace(-1 * np.ceil(max_curr / 10 ** utils.get_exponent(max_curr)),
                                 np.ceil(max_curr / 10 ** utils.get_exponent(max_curr)), num=5)
                     * 10 ** utils.get_exponent(max_curr))
    ax_iv.set_yticklabels(np.linspace(-1 * np.ceil(max_curr / 10 ** utils.get_exponent(max_curr)),
                                      np.ceil(max_curr / 10 ** utils.get_exponent(max_curr)), num=5))
    ax_iv.set_xticks(np.linspace(np.round(min(iv_bias_1)), np.round(max(iv_bias_1), 1), num=iv_num_xticks))

    if plot_mean_current:
        for i in range(len(hold_trace.bias_steps)):
            ax_iv.plot(hold_trace.bias_steps[i], avg_current_on_step[i], marker='o', ls='', ms=2.5,
                       markeredgecolor='k', markeredgewidth=0.2, c=color_list[i])

            ax_noise_1.plot(hold_trace.bias_steps[i], hold_trace_areas[i], marker='o', ls='', ms=2.5,
                            markeredgecolor='k', markeredgewidth=0.2, c=color_list[i])
            ax_noise_2.plot(hold_trace.bias_steps[i], hold_trace_current_noise[i], marker='o', ls='', ms=2.5,
                            markeredgecolor='k', markeredgewidth=0.2, c=color_list[i])

    ax_hold, par_hold = hold_trace.plot_hold_traces(direction=direction,
                                                    plot_step_ranges=False,
                                                    plot_psd_intervals=False,
                                                    conductance=False,
                                                    ax=ax_hold, log_scale_y=False,
                                                    ax_colors=accent_colors,
                                                    smoothing=smoothing)

    ax_hold.set_ylim(-1 * max_curr, max_curr)
    ax_hold.set_yticks(np.linspace(-1 * np.ceil(max_curr / 10 ** utils.get_exponent(max_curr)),
                                   np.ceil(max_curr / 10 ** utils.get_exponent(max_curr)), num=5)
                       * 10 ** utils.get_exponent(max_curr))
    ax_hold.set_yticklabels(np.linspace(-1 * np.ceil(max_curr / 10 ** utils.get_exponent(max_curr)),
                                        np.ceil(max_curr / 10 ** utils.get_exponent(max_curr)), num=5))

    if utils.get_exponent(max_curr) == -6:
        ax_hold.set_ylabel(r'Current [$10^{-6}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-6}\;\mathrm{A}$]', labelpad=0.5)
    elif utils.get_exponent(max_curr) == -7:
        ax_hold.set_ylabel(r'Current [$10^{-7}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-7}\;\mathrm{A}$]', labelpad=0.5)
    elif utils.get_exponent(max_curr) == -8:
        ax_hold.set_ylabel(r'Current [$10^{-8}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-8}\;\mathrm{A}$]', labelpad=0.5)
    elif utils.get_exponent(max_curr) == -9:
        ax_hold.set_ylabel(r'Current [$10^{-9}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-9}\;\mathrm{A}$]', labelpad=0.5)
    elif utils.get_exponent(max_curr) == -10:
        ax_hold.set_ylabel(r'Current [$10^{-10}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-10}\;\mathrm{A}$]', labelpad=0.5)
    elif utils.get_exponent(max_curr) == -5:
        ax_hold.set_ylabel(r'Current [$10^{-5}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-5}\;\mathrm{A}$]', labelpad=0.5)
    elif utils.get_exponent(max_curr) == -4:
        ax_hold.set_ylabel(r'Current [$10^{-4}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-4}\;\mathrm{A}$]', labelpad=0.5)
    else:
        raise UserWarning(f'No axis label defined for this case. Refer to plots.py line 681, to add a label'
                          f'for the case when utils.get_exponent(max_curr) = {utils.get_exponent(max_curr)}')

    # ax_hold.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax_hold.set_xlabel('Time [s]')

    bias_vals = np.concatenate((hold_trace.bias_steps, np.array([max(iv_bias_1)])))

    par_hold.set_yticks(np.linspace(np.round(min(iv_bias_1)), np.round(max(iv_bias_1)),
                                    num=iv_num_xticks))
    par_hold.set_yticklabels(np.linspace(np.round(min(iv_bias_1)), np.round(max(iv_bias_1)),
                                         num=iv_num_xticks))
    par_hold.set_ylabel('Bias [V]')

    for i, interval in enumerate(psd_intervals):
        ax_hold.axvspan(interval[0] / 50_000, interval[-1] / 50_000, color=cols[i], ec=None, alpha=0.5)

        for j in interval:
            ax_hold.axvline(j / 50_000, ls='--', c=vline_color, lw=0.6)

    ax_psd = hold_trace.plot_psds(ax=ax_psd, pull=pull, plot_legend=False, which_psds=which_psds, plot_guides=False,
                                  color_list=color_list)

    ax_noise_1.set_xlabel('Bias [V]')
    ax_noise_2.set_xlabel('Bias [V]')

    ax_noise_1.set_ylabel(r'$\Delta I^2 [\mathrm{A}^2]$')
    ax_noise_2.set_ylabel(r'$\Delta I/I$')

    ax_noise_1.set_xscale('log')
    ax_noise_1.set_yscale('log')
    ax_noise_2.set_xscale('log')
    ax_noise_2.set_yscale('log')

    ax_noise_1.xaxis.tick_top()
    ax_noise_1.xaxis.set_label_position('top')
    ax_noise_1.xaxis.set_ticks_position('both')
    # ax_noise_1.yaxis.tick_right()
    # ax_noise_1.yaxis.set_label_position('right')
    ax_noise_1.yaxis.set_ticks_position('both')

    ax_noise_2.yaxis.tick_right()
    ax_noise_2.yaxis.set_label_position('right')
    ax_noise_2.xaxis.tick_top()
    ax_noise_2.xaxis.set_label_position('top')
    ax_noise_2.xaxis.set_ticks_position('both')
    ax_noise_2.yaxis.set_ticks_position('both')

    ax_noise_1.set_ylim(10 ** min(np.floor(np.log10(hold_trace_areas[hold_trace.bias_steps > 0]))),
                        10 ** max(np.ceil(np.log10(hold_trace_areas))))

    ax_noise_1.tick_params(axis='x', labelsize=4)
    ax_noise_1.tick_params(axis='y', labelsize=4)
    ax_noise_2.tick_params(axis='x', labelsize=4)
    ax_noise_2.tick_params(axis='y', labelsize=4)

    return fig, ax_trace, ax_hold, par_hold, ax_iv, ax_psd

hold_trace = HoldTrace(iv_good_pull[0],
                       load_from=home_folder, bias_offset=0,
                       r_serial_ohm=100_100, min_step_len=4_000, min_height=1)

selected_blues = colormaps['Blues'](np.linspace(0.2, 1, np.unique(hold_trace.bias_steps)[1:].shape[0]))

my_blues = np.vstack((selected_blues[0].reshape(1, -1),
                      np.array([[0, 0, 0, 1]]).reshape(1, -1),
                      selected_blues,
                      selected_blues[10].reshape(1, -1),
                      selected_blues[5].reshape(1, -1),
                      selected_blues[0].reshape(1, -1),
                      np.array([[0, 0, 0, 1]]).reshape(1, -1)))

selected_reds = colormaps['Reds'](np.linspace(0.2, 1, np.unique(hold_trace.bias_steps)[1:].shape[0]))

my_reds = np.vstack((selected_reds[0].reshape(1, -1),
                     np.array([[0, 0, 0, 1]]).reshape(1, -1),
                     selected_reds,
                     selected_reds[10].reshape(1, -1),
                     selected_reds[5].reshape(1, -1),
                     selected_reds[0].reshape(1, -1),
                     np.array([[0, 0, 0, 1]]).reshape(1, -1)))

for which_trace in tqdm(iv_good_pull, desc='Saving pull plots'):
    fig, _, _, _, _, _ = plot_half_ivs(trace_num=which_trace,
                                       direction='pull',
                                       main_colors=('cornflowerblue', 'indianred'),
                                       accent_colors=('royalblue', 'firebrick'),
                                       vline_color='grey',
                                       color_list=my_blues,
                                       smoothing=1,
                                       iv_num_xticks=5,
                                       which_psds=None,
                                       plot_mean_current=True,
                                       fig_size=utils.cm2inch(12, 8))

    fig.suptitle(f'Trace {which_trace}', fontsize=6)

    plt.savefig(home_folder.joinpath(f'results/IVs/figs/pull_{which_trace}.png'), bbox_inches='tight')

    plt.clf()
    plt.close(fig='all')

for which_trace in tqdm(iv_good_push, desc='Saving pull plots'):
    fig, _, _, _, _, _ = plot_half_ivs(trace_num=which_trace,
                                       direction='push',
                                       main_colors=('cornflowerblue', 'indianred'),
                                       accent_colors=('royalblue', 'firebrick'),
                                       vline_color='grey',
                                       color_list=my_reds,
                                       smoothing=1,
                                       iv_num_xticks=5,
                                       which_psds=None,
                                       plot_mean_current=True,
                                       fig_size=utils.cm2inch(12, 8))

    fig.suptitle(f'Trace {which_trace}', fontsize=6)

    plt.savefig(home_folder.joinpath(f'results/IVs/figs/push_{which_trace}.png'), bbox_inches='tight')

    plt.clf()
    plt.close(fig='all')
