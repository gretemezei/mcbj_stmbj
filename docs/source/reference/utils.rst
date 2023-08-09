.. _utils

====
Utility functions
====

.. currentmodule:: utils

Converters
~~~~~~~~~~

.. autosummary::
    :toctree: api/

    convert_g0_to_ohm
    convert_ohm_to_g0
    convert_pt_to_sec
    convert_sec_to_pt
    convert_to_block_and_trace_num
    convert_to_trace
    convert_current_noise_to_conductance

    cm2inch
    point2inch

Calculations
~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    calculate_conductance_g0
    calculate_resistance_r0
    calculate_resistance_ohm
    get_exponent
    moving_average
    log_avg
    round_bias_steps
    find_level
    calc_hist_1d_single
    calc_hist_2d_single
    interpolate
    align_trace
    calc_covariance
    calc_correlation
    largest_divisor
    custom_error

File handling
~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    collect_traces
    load_block_to_list
    read_waveform_params
    get_name_from_path
    get_num_from_name
    choose_files
    choose_traces
    sort_by_trace_num
    load_scopedata

Fit Functions
~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    fit_func_lin
    gaussian_fun

Miscellaneous
~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    execute_and_measure_time
    check_date
    count_bool_groups
    even_ext
