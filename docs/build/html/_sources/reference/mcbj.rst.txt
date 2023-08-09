.. _mcbj

====
Break Junction and Hold measurement analysis
====

.. currentmodule:: mcbj

Break Junction Trace Pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    TracePair
    TracePair.load_trace_pair
    TracePair.calc_plateau_length
    TracePair.plot_trace_pair

.. currentmodule:: mcbj

Break Junction Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Histogram
    Histogram.calc_hist_1d
    Histogram.calc_hist_2d
    Histogram.calc_plateau_length_hist
    Histogram.calc_temporal_hist
    Histogram.calc_stats
    Histogram.calc_corr_hist_2d

    Histogram.plot_example_traces
    Histogram.plot_hist_1d
    Histogram.plot_hist_2d_both
    Histogram.plot_hist_2d_one
    Histogram.plot_plateau_length_hist
    Histogram.plot_temporal_hist
    Histogram.plot_corr

    Histogram.save_histogram
    Histogram.load_histogram

.. currentmodule:: mcbj

Hold Trace Analysis
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

     HoldTrace
     HoldTrace.load_hold_traces
     HoldTrace.plot_hold_traces
     HoldTrace.analyse_hold_trace
     HoldTrace.area_under_psds
     HoldTrace.area_under_single_psd
     HoldTrace.calc_exponents4psds
     HoldTrace.calc_noise_value
     HoldTrace.calculate_avg_value_on_step
     HoldTrace.calculate_freq_resolution
     HoldTrace.calculate_psds
     HoldTrace.find_bias_steps
     HoldTrace.find_max_fft_interval
     HoldTrace.find_psd_intervals
     HoldTrace.find_signal_steps
     HoldTrace.fit_line_to_psd
     HoldTrace.is_long_enough
     HoldTrace.plot_psds
     HoldTrace.remove_thermal_background_1d
     HoldTrace.remove_thermal_background_2d

.. currentmodule:: mcbj

Noise Analysis
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

     NoiseStats
     NoiseStats.calc_stats
     NoiseStats.calc_stats_old
     NoiseStats.plot_correlation
     NoiseStats.plot_noise_power_2dhist
     NoiseStats.scat_and_hist
     NoiseStats.scatterplot
     NoiseStats.scatterplot_pull_push