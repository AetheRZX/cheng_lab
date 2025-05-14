import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import scipy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches 
import numpy as np
import math
import scipy.signal as signal

from matplotlib.widgets import SpanSelector

def find_steady_peak_threshold(axial_distance, data,
                               peak_height_fraction=0.85,
                               smoothing_window=21, smoothing_order=2,
                               min_width_points=10):
    """
    Finds the steady portion based on proximity to the peak height.

    Parameters:
        axial_distance (np.ndarray): Axial distance data.
        data (np.ndarray): Data to analyze.
        peak_height_fraction (float): Fraction of peak height to use as threshold (e.g., 0.9 for 90%).
        smoothing_window (int): Window size for Savitzky-Golay filter for peak finding.
        smoothing_order (int): Polynomial order for Savitzky-Golay filter.
        min_width_points (int): Minimum number of points required for a valid steady region.

    Returns:
        dict or None: Dictionary with 'steady_start_idx', 'steady_end_idx',
                      'steady_average', etc., or None if no valid region found.
    """
    if len(data) < smoothing_window:
        print("Warning: Data length is smaller than smoothing window.")
        # Optionally handle this case, e.g., return None or skip smoothing
        smoothed_data = data
    else:
        smoothed_data = savgol_filter(data, window_length=smoothing_window, polyorder=smoothing_order)

    # Find the peak within the smoothed data
    try:
        peak_idx = np.argmax(smoothed_data)
        peak_val = smoothed_data[peak_idx]
    except ValueError: # Handle empty or all-NaN data
        print("Error: Could not find peak in data.")
        return None


    # Define the threshold based on the peak value
    threshold = peak_val * peak_height_fraction

    # Find all indices where the smoothed data is above the threshold
    above_threshold_indices = np.where(smoothed_data >= threshold)[0]

    if len(above_threshold_indices) == 0:
        print(f"No points found above {peak_height_fraction*100:.1f}% of peak height.")
        return None

    # Find the contiguous block of indices that includes the peak index
    # Split the indices into contiguous blocks
    blocks = np.split(above_threshold_indices, np.where(np.diff(above_threshold_indices) != 1)[0]+1)

    steady_block = None
    for block in blocks:
        if block[0] <= peak_idx <= block[-1]:
            steady_block = block
            break

    if steady_block is None or len(steady_block) < min_width_points :
        print(f"No suitable contiguous block found around peak (min width: {min_width_points}).")
        return None

    steady_start_idx = steady_block[0]
    steady_end_idx = steady_block[-1] # This is INCLUSIVE

    # Calculate average using ORIGINAL data in the found range
    steady_data_orig = data[steady_start_idx : steady_end_idx + 1]
    steady_axial_orig = axial_distance[steady_start_idx : steady_end_idx + 1]
    steady_average = np.mean(steady_data_orig)

    return {
        "steady_start_idx": steady_start_idx,
        "steady_end_idx": steady_end_idx + 1, # Make exclusive for consistency with slicing
        "steady_average": steady_average,
        "steady_axial": steady_axial_orig,
        "steady_data": steady_data_orig,
        "peak_idx": peak_idx,
        "threshold": threshold
    }


# --- Helper Function for Dynamic Limits ---
def calculate_dynamic_limits(data_arrays, padding_factor=0.05, round_multiple=None, zero_baseline=False):
    if not isinstance(data_arrays, list):
        data_arrays = [data_arrays]
    all_data = np.concatenate([arr[np.isfinite(arr)] for arr in data_arrays if hasattr(arr, 'size') and arr.size > 0])
    if all_data.size == 0:
        return (0, 1) if not zero_baseline else (-0.5, 0.5)
    data_min, data_max = np.min(all_data), np.max(all_data)
    if np.isclose(data_min, data_max):
        delta = abs(data_min) * 0.1 if not np.isclose(data_min, 0) else 0.1
        data_range = delta * 2; data_min -= delta; data_max += delta
    else: data_range = data_max - data_min
    padding = data_range * padding_factor
    lim_min, lim_max = data_min - padding, data_max + padding
    if zero_baseline:
        if lim_min > 0: lim_min = 0
        elif lim_max < 0: lim_max = 0
    if round_multiple is not None and round_multiple > 0:
        lim_min = math.floor(lim_min / round_multiple) * round_multiple
        lim_max = math.ceil(lim_max / round_multiple) * round_multiple
        if np.isclose(lim_max, lim_min):
             lim_min -= round_multiple / 2; lim_max += round_multiple / 2
    return lim_min, lim_max

def get_manual_indices_interactive(dataframe, title_prefix="Select Data Region"):
    """
    Displays a simplified plot with Load on left Y, Gaps on right Y,
    and allows the user to select a start and end index by drawing a
    horizontal span on Axial Distance.

    Args:
        dataframe (pl.DataFrame): The Polars DataFrame containing the data.
                                  Must contain "Axial Distance [mm]", "Load [N]",
                                  "Left Gap [μm]", and "Right Gap [μm]".
        title_prefix (str): Prefix for the plot window title.

    Returns:
        tuple: (start_index, end_index) or (None, None) if selection is aborted.
    """
    # --- Define column names ---
    x_col = "Axial Distance [mm]"
    load_col = "Load [N]"
    left_gap_col = "Left Gap [μm]"
    right_gap_col = "Right Gap [μm]"

    # --- Check if necessary columns exist ---
    required_cols = [x_col, load_col, left_gap_col, right_gap_col]
    for col in required_cols:
        if col not in dataframe.columns:
            print(f"ERROR: Required column '{col}' not found in DataFrame. Cannot create plot.")
            return None, None

    # --- Prepare data ---
    x_plot_data = dataframe[x_col].to_numpy()
    load_data = dataframe[load_col].to_numpy()
    left_gap_data = dataframe[left_gap_col].to_numpy()
    right_gap_data = dataframe[right_gap_col].to_numpy()

    x_data_for_indices = x_plot_data  # For np.searchsorted

    # --- Create Figure and Axes ---
    fig, ax_load = plt.subplots(figsize=(12, 6)) # Main axes for Load
    ax_gaps = ax_load.twinx()                    # Twin axes for Gaps

    # --- Plotting ---
    line_load, = ax_load.plot(x_plot_data, load_data, color='black', label=load_col)
    line_left_gap, = ax_gaps.plot(x_plot_data, left_gap_data, color='blue', label=left_gap_col, linestyle='--')
    line_right_gap, = ax_gaps.plot(x_plot_data, right_gap_data, color='red', label=right_gap_col, linestyle=':')

    # --- Axis Labels ---
    ax_load.set_xlabel(x_col)
    ax_load.set_ylabel(load_col, color='black')
    ax_gaps.set_ylabel("Gap Values (μm)", color='dimgray') # General label for the right axis

    # --- Tick Colors ---
    ax_load.tick_params(axis='y', labelcolor='black')
    ax_gaps.tick_params(axis='y', labelcolor='dimgray')

    # --- Legend ---
    # Collect lines and labels from both axes for a single legend
    lines = [line_load, line_left_gap, line_right_gap]
    labels = [l.get_label() for l in lines]
    ax_load.legend(lines, labels, loc='best')

    # --- Title ---
    # Using fig.suptitle for the main instruction, ax_load.set_title for selection feedback
    fig.suptitle(f"{title_prefix}\nDrag on '{x_col}' to select. Close window when done.", fontsize=12)
    ax_load.set_title("", fontsize=10) # Placeholder for selection feedback
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    # --- SpanSelector Logic ---
    selected_region_indices = {"start": None, "end": None}

    def onselect(xmin_sel, xmax_sel):
        start_idx = np.searchsorted(x_data_for_indices, xmin_sel, side='left')
        end_idx = np.searchsorted(x_data_for_indices, xmax_sel, side='right')
        start_idx = max(0, start_idx)
        end_idx = min(len(x_data_for_indices), end_idx)

        if start_idx < end_idx:
            selected_region_indices["start"] = start_idx
            selected_region_indices["end"] = end_idx
            feedback_msg = f"Selected: Idx {start_idx}-{end_idx-1} (X: {xmin_sel:.2f} to {xmax_sel:.2f})"
            ax_load.set_title(feedback_msg, color='green', fontsize=10)
            print(feedback_msg)
            fig.canvas.draw_idle()
        else:
            ax_load.set_title("Selection invalid. Try again.", color='red', fontsize=10)
            print("Selection too small or invalid.")
            fig.canvas.draw_idle()

    span = SpanSelector(ax_load, onselect, 'horizontal', useblit=False,
                        props=dict(alpha=0.3, facecolor='lightgray'), # Simple rectangle for selection
                        button=1)

    fig._span_selector_ref = span

    plt.show(block=True)

    if selected_region_indices["start"] is not None and selected_region_indices["end"] is not None:
        print(f"Final selection: Start Index = {selected_region_indices['start']}, End Index = {selected_region_indices['end']}")
        return selected_region_indices["start"], selected_region_indices["end"]
    else:
        print("No valid selection made, or window closed before selection.")
        return None, None

