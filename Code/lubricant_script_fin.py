from helper_func import find_steady_peak_threshold, calculate_dynamic_limits, get_manual_indices_interactive
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

# Load the file
### These 2 are problematic as it could not detect the edges
# data_file = Path("../Data/0.8mpers-60n-0.8pas.xlsx") # Change this to the appropriate folder/file
data_file = Path(r"E:\School\Year 3\cheng_lab\Data\0.3mpers-110n-11.3pas.xlsx") 

# data_file = Path("../Data/1.5mpers-30n-11.3pas.xlsx") 
# data_file = Path("../Data/3kn-0.5.xlsx")
# data_file = Path(r"..\Data\1mpers-20n-0.8pas.xlsx")
# df = pl.read_excel(source=data_file, sheet_name="Analog")

df = pl.read_excel(source=data_file, sheet_name="Analog #1")

## Parameter you can change
sampling_speed = 1000
speed = 1.5
data_point_length = speed / sampling_speed


#### Below this you can leave it be
## Process the file
header_names = df.columns
x_data = pl.Series("Axial Distance [mm]", np.arange(0, df.shape[0]) * data_point_length) * 1000
df = df.with_columns(
    x_data,
    ((16+pl.col("Voltage")) * 1000).alias("Left Gap [\u03bcm]"),
    ((16+((pl.col("Dev1/ai3")/250*1000)-4)*(10/16))*1000).alias("Right Gap [\u03bcm]"),
    (pl.col("Dev1/ai2")*112.3898+558.30904).alias("Load [N]")
)

bound = 1000
left_gap = df["Left Gap [μm]"].to_numpy()
peaks, _ = find_peaks(left_gap)

peaks = sorted(peaks, key=lambda x: left_gap[x], reverse=True)
outlier_index = np.sort(peaks[:2]) # Sort ensures the 0 index is the left point 

start_index, end_index = outlier_index[0] + bound, outlier_index[1] - bound * 3 # Take the bound away from the file

df = df.slice(start_index, end_index) 

x_data = pl.Series("Axial Distance [mm]", np.arange(0, df.shape[0]) * data_point_length) * 1000

N = 2**10
fs = 1000
signal_data = df["Left Gap [\u03bcm]"]
signal_data = np.asarray(signal_data, dtype=np.float64)
right_signal = df["Right Gap [\u03bcm]"]

# Design a High-Pass Butterworth Filter
cutoff_freq = 5   # Remove frequencies below 5 Hz
order = 4
b, a = signal.butter(order, cutoff_freq / (fs / 2), btype='high')
# Apply the filter
filtered_signal = signal.filtfilt(b, a, signal_data)
filtered_right_signal = signal.filtfilt(b, a, right_signal)


# Detect positive and negative peaks
left_peaks, _ = find_peaks(filtered_signal, height=20)  # adjust height
left_troughs, _ = find_peaks(-filtered_signal, height=20)
right_peaks, _ = find_peaks(filtered_right_signal, height=20)
right_troughs, _ = find_peaks(-filtered_right_signal, height=20)

# Combine peaks and troughs for both
left_spikes = np.sort(np.concatenate([left_peaks, left_troughs]))
right_spikes = np.sort(np.concatenate([right_peaks, right_troughs]))

# Compare: check for matching spikes within a small lag tolerance
tolerance = 5  # samples
matches = []

for i in left_spikes:
    if np.any(np.abs(right_spikes - i) <= tolerance):
        matches.append(i)


# Parameters
window_size = 500  # Number of samples per window (adjust if needed)

# Count matched spikes in sliding windows
match_counts = []
window_starts = range(0, len(filtered_signal) - window_size + 1, 50)  # Slide every 50 samples

for start in window_starts:
    end = start + window_size
    count = np.sum((np.array(matches) >= start) & (np.array(matches) < end))
    match_counts.append(count)

# Find the window with the most matched spikes
best_window_index = np.argmax(match_counts)
best_start = window_starts[best_window_index]
best_end = best_start + window_size

print(f"Most active window: {best_start} to {best_end} with {match_counts[best_window_index]} matched spikes.")

# Plot zoomed-in region
plt.figure(figsize=(12, 4))
plt.plot(range(best_start, best_end), filtered_signal[best_start:best_end], label="Left Filtered", alpha=0.6)
plt.plot(range(best_start, best_end), filtered_right_signal[best_start:best_end], label="Right Filtered", alpha=0.6)

# Plot only matched spikes in this region
matched_in_window = [i for i in matches if best_start <= i < best_end]
plt.scatter(matched_in_window,
            filtered_signal[matched_in_window],
            color='red', label="Matched Spikes")

plt.title("Zoomed View: Region with Most Coincidence Spikes")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# === Parameters for Bump Detection ===
window_size = 100
step_size = 10
min_spike_count_density = 5 # Threshold for find_peaks on density
peak_prominence = 10       # Prominence for find_peaks on density
min_peak_distance_samples = 200
density_boundary_fraction = 0.2 # Fraction of peak density height for boundary search

# === 1. Calculate Spike Density (Unchanged) ===
match_counts = []
matches = np.array(matches) # Ensure numpy array
window_starts = range(0, len(filtered_signal) - window_size, step_size)
window_centers = []

for start in window_starts:
    end = start + window_size
    count = np.sum((matches >= start) & (matches < end))
    match_counts.append(count)
    window_centers.append(start + window_size // 2)

match_counts = np.array(match_counts)
window_centers = np.array(window_centers)

# === 2. Find ALL Peaks in Density Initially (Unchanged) ===
min_peak_distance_windows = int(np.ceil(min_peak_distance_samples / step_size))
density_peaks_indices, properties = find_peaks(
    match_counts,
    height=min_spike_count_density,
    prominence=peak_prominence,
    distance=min_peak_distance_windows
)

print(f"Found {len(density_peaks_indices)} potential density peaks initially.")

# === 3. Calculate Boundaries and Actual Spike Counts for ALL Peaks ===
preliminary_bumps = [] # Store info before sorting and overlap check

if len(density_peaks_indices) > 0:
    peak_heights = properties.get('peak_heights', None) # Get heights if available

    for i, peak_idx in enumerate(density_peaks_indices):
        peak_height = match_counts[peak_idx] # Use actual count at peak index

        # Find boundaries based on density drop
        left_idx = peak_idx
        boundary_threshold = peak_height * density_boundary_fraction
        while left_idx > 0 and match_counts[left_idx - 1] >= boundary_threshold:
             left_idx -= 1
        right_idx = peak_idx
        while right_idx < len(match_counts) - 1 and match_counts[right_idx + 1] >= boundary_threshold:
             right_idx += 1

        # Define search region and find spikes
        search_start_signal = window_starts[left_idx]
        search_end_signal = window_starts[right_idx] + window_size
        spikes_in_bump_region = matches[(matches >= search_start_signal) & (matches < search_end_signal)]
        num_spikes_in_region = len(spikes_in_bump_region)

        # Store preliminary info ONLY if spikes were actually found
        if num_spikes_in_region > 0:
            bump_start_idx = np.min(spikes_in_bump_region)
            bump_end_idx = np.max(spikes_in_bump_region) # Inclusive end
            bump_center_idx = int(np.median(spikes_in_bump_region))

            preliminary_bumps.append({
                 'start': bump_start_idx,
                 'end': bump_end_idx + 1, # Make exclusive
                 'center': bump_center_idx,
                 'num_spikes': num_spikes_in_region, # The crucial metric
                 'peak_height': peak_height, # Original density peak height
                 'density_peak_idx': peak_idx, # Original density peak index
                 'search_start': search_start_signal, # Keep for debugging maybe
                 'search_end': search_end_signal
             })
        else:
            print(f" -> Density Peak at index {peak_idx} (height {peak_height:.1f}) yielded no actual spikes in region {search_start_signal}-{search_end_signal}.")

else:
    print("No density peaks found meeting the initial criteria.")


# === 4. Sort Preliminary Bumps by Actual Spike Count ===
if preliminary_bumps:
    # Sort by 'num_spikes' in descending order
    sorted_preliminary_bumps = sorted(preliminary_bumps, key=lambda item: item['num_spikes'], reverse=True)
    print(f"\nProcessed {len(sorted_preliminary_bumps)} potential bumps with actual spikes. Now checking overlaps...")
else:
    sorted_preliminary_bumps = []
    print("\nNo potential bumps with actual spikes found.")

# === 5. Filter Overlapping Bumps (Prioritizing higher spike counts) ===
detected_bumps = [] # This will be the final list

for potential_bump in sorted_preliminary_bumps:
    is_overlapping = False
    # Compare potential_bump against bumps already added to detected_bumps
    for existing_bump in detected_bumps:
        # Check for overlap: max(starts) < min(ends)
        # Use potential_bump['end'] (exclusive) and existing_bump['end'] (exclusive)
        if max(potential_bump['start'], existing_bump['start']) < min(potential_bump['end'], existing_bump['end']):
             is_overlapping = True
             print(f" -> Bump {potential_bump['start']}-{potential_bump['end']-1} ({potential_bump['num_spikes']} spikes) overlaps with kept bump {existing_bump['start']}-{existing_bump['end']-1}. Skipping.")
             break # Stop checking overlaps for this potential_bump

    # If it doesn't overlap with any already kept bumps, add it
    if not is_overlapping:
        detected_bumps.append(potential_bump)
        print(f" -> Keeping Bump: {potential_bump['start']}-{potential_bump['end']-1} ({potential_bump['num_spikes']} spikes)")


# --- Final Result ---
# 'detected_bumps' now contains non-overlapping bumps.
# Because we processed the sorted list and added non-overlapping ones,
# the bump with the globally highest spike count (among non-overlapping ones)
# should naturally be at index 0.

if detected_bumps:
    print(f"\nFinal detected non-overlapping bumps: {len(detected_bumps)}")
    # The first element IS the best bump based on spike count due to sorting before filtering
    print(f"Best bump (Index 0): Start={detected_bumps[0]['start']}, End={detected_bumps[0]['end']-1}, Center={detected_bumps[0]['center']}, Num Spikes={detected_bumps[0]['num_spikes']}")
    
    start_index, end_index = detected_bumps[0]["start"], detected_bumps[0]["end"]
    # Optionally print other bumps
    # for i, bump in enumerate(detected_bumps[1:], 1):
    #    print(f"Bump {i+1}: Start={bump['start']}, End={bump['end']-1}, Center={bump['center']}, Num Spikes={bump['num_spikes']}")
else:
    print("\nNo valid, non-overlapping bumps were detected. going to manual mode")
    # Call manual mode here
    start_index, end_index = get_manual_indices_interactive(df)

# --- Main Analysis Loop ---
analysis_successful = False

while not analysis_successful:
    # --- Step 1: Get/Update Selection Indices ---
    if start_index is None or end_index is None: # Or if a retry is explicitly needed
        print("\n--- Entering Manual Selection Mode ---")
        # Call your interactive selection function (ensure it's defined)
        # For this example, I'm using the simplified one.
        s_idx, e_idx = get_manual_indices_interactive(df, title_prefix="Select Region for Analysis")

        if s_idx is None or e_idx is None:
            print("User cancelled selection. Exiting analysis.")
            break # Exit the while loop entirely
        
        # Validate selection before assigning
        if e_idx <= s_idx:
            print(f"Invalid selection: end_index ({e_idx}) must be greater than start_index ({s_idx}). Please try again.")
            # Reset start/end_index to None to re-trigger selection in the next loop iteration
            start_index, end_index = None, None
            continue # Go to the start of the while loop
        
        start_index, end_index = s_idx, e_idx
        print(f"Region selected: Indices {start_index} to {end_index-1}")


    # --- Step 2: Prepare Data Based on Current start_index and end_index ---
    print(f"\nProcessing data for selected indices: {start_index} to {end_index-1}")
    
    # Calculate average_force (e.g., from the start of the original df)
    # Ensure there are enough points for the mean calculation
    avg_force_calc_len = min(1000, len(df))
    average_force = 0
    if avg_force_calc_len > 0:
        average_force = np.mean(df["Load [N]"][:avg_force_calc_len].to_numpy())

    # Reference values for subtraction from the original df at the current start_index
    # These ensure that even if we re-select, the "zeroing" is relative to the new start
    try:
        ref_axial_dist = df["Axial Distance [mm]"].item(start_index)
        ref_left_gap = df["Left Gap [μm]"].item(start_index)
        ref_right_gap = df["Right Gap [μm]"].item(start_index)
    except IndexError:
        print(f"Error: start_index {start_index} is out of bounds for the DataFrame (length {len(df)}).")
        print("Please try selecting a valid region.")
        start_index, end_index = None, None # Force re-selection
        continue


    # Create a temporary DataFrame for the *selected slice* and then apply transformations
    # This avoids modifying the global `df` or `df_filtered` repeatedly in a confusing way.
    df_slice = df[start_index:end_index]

    if len(df_slice) == 0:
        print("The selected slice is empty. Please select a valid region.")
        start_index, end_index = None, None # Force re-selection
        continue

    # Apply transformations to the slice
    x_axis = (df_slice["Axial Distance [mm]"] - ref_axial_dist).to_numpy()
    left_gap = (df_slice["Left Gap [μm]"] - ref_left_gap).to_numpy()
    right_gap = (df_slice["Right Gap [μm]"] - ref_right_gap).to_numpy()
    load_data = (df_slice["Load [N]"] - average_force).to_numpy()


    # --- Step 3: Analyze Steady Portions ---
    print("Analyzing steady portions...")
    result_left = find_steady_peak_threshold(x_axis, left_gap)
    result_right = find_steady_peak_threshold(x_axis, right_gap)

    # --- Step 4: Check Results and Decide Next Action ---
    if result_left is None or result_right is None:
        print("\n" + "="*30)
        print("Analysis Error: Could not find suitable steady portions for the selected region.")
        if result_left is None: print(" -> Problem with Left Gap.")
        if result_right is None: print(" -> Problem with Right Gap.")
        print("="*30)
        
        user_choice = input("Would you like to try selecting a different region? (y/n): ").lower()
        if user_choice == 'y':
            start_index, end_index = None, None # Reset to trigger manual_mode in the next loop
            print("Returning to manual selection mode...")
            # The loop will continue, and the condition at the start will re-trigger manual selection
        else:
            print("Exiting analysis.")
            break # Exit the while loop
    else:
        # Success! Process the results
        try:
            left_gap_avg = result_left["steady_average"]
            right_gap_avg = result_right["steady_average"]
            print(f"\nAnalysis Successful for selected region:")
            print(f"  Left Gap Average: {left_gap_avg:.2f}")
            print(f"  Right Gap Average: {right_gap_avg:.2f}")
            analysis_successful = True # Set flag to exit the loop

            # # --- Step 5: Plotting (if successful) ---
            # plt.figure(figsize=(10, 6))
            # plt.plot(x_axis, left_gap, label=f"Left Gap (avg: {left_gap_avg:.2f})")
            # plt.plot(x_axis, right_gap, label=f"Right Gap (avg: {right_gap_avg:.2f})")
            # plt.plot(x_axis, load_data, label="Load (Processed)")
            # plt.axhline(left_gap_avg, color='blue', linestyle=':', alpha=0.7)
            # plt.axhline(right_gap_avg, color='red', linestyle=':', alpha=0.7)
            # plt.xlabel("Processed Axial Distance [mm]")
            # plt.ylabel("Values")
            # plt.title(f"Analyzed Region (Original Indices: {start_index}-{end_index-1})")
            # plt.legend()
            # plt.grid(True)
            # plt.show()

        except KeyError as e:
            print(f"Error: Key '{e}' not found in results from find_steady_peak_threshold.")
            print("Please check the structure of the dictionary returned by that function.")
            user_choice = input("Would you like to try selecting a different region? (y/n): ").lower()
            if user_choice == 'y':
                start_index, end_index = None, None
            else:
                print("Exiting analysis.")
                break
        except Exception as e: # Catch any other unexpected error during result processing
            print(f"An unexpected error occurred: {e}")
            user_choice = input("Would you like to try selecting a different region? (y/n): ").lower()
            if user_choice == 'y':
                start_index, end_index = None, None
            else:
                print("Exiting analysis.")
                break

# --- End of While Loop ---

if analysis_successful:
    print("\nProcessing finished successfully.")
else:
    print("\nProcessing was not completed or was aborted by the user.")



# --- Calculate Dynamic Limits ---
# Setting explicit rounding multiples based on the target image's style
x_lim_min, x_lim_max = calculate_dynamic_limits([x_axis], padding_factor=0.05, round_multiple=20.0)
y_load_lim_min, y_load_lim_max = calculate_dynamic_limits([load_data], padding_factor=0.05, round_multiple=10.0, zero_baseline=True)
y_gap_lim_min, y_gap_lim_max = calculate_dynamic_limits([left_gap, right_gap], padding_factor=0.05, round_multiple=20.0) # Adjust y-gap if needed

# --- Choose Tick Intervals ---
x_tick_interval = 20.0
y_load_tick_interval = 10.0
y_gap_tick_interval = 20.0 # Match y-gap rounding

# --- Determine Bounding Box Coordinates based on steady indices and data ---
box_defined = False
box_y_padding_factor = 0.1 # Padding for the box height

if ('result_left' in locals() and "steady_start_idx" in result_left and "steady_end_idx" in result_left and
    'result_right' in locals() and "steady_start_idx" in result_right and "steady_end_idx" in result_right): # Check both results

    steady_start_idx_l = result_left["steady_start_idx"]
    steady_end_idx_l = result_left["steady_end_idx"] # Exclusive
    steady_start_idx_r = result_right["steady_start_idx"]
    steady_end_idx_r = result_right["steady_end_idx"] # Exclusive

    # Check if indices are valid for BOTH left and right (or adjust logic if only one is needed)
    if (steady_start_idx_l < len(x_axis) and steady_end_idx_l <= len(x_axis) and steady_end_idx_l > steady_start_idx_l and
        steady_start_idx_r < len(x_axis) and steady_end_idx_r <= len(x_axis) and steady_end_idx_r > steady_start_idx_r):

        # --- Calculate X extent from left indices (or union/intersection if preferred) ---
        box_x_start_val = x_axis[steady_start_idx_l]
        box_x_end_val = x_axis[steady_end_idx_l - 1] # Last included point
        box_width = box_x_end_val - box_x_start_val

        # --- Calculate Y extent dynamically from data within steady ranges ---
        steady_y_left = left_gap[steady_start_idx_l:steady_end_idx_l]
        steady_y_right = right_gap[steady_start_idx_r:steady_end_idx_r]

        if steady_y_left.size > 0 and steady_y_right.size > 0:
            y_min_in_range = min(np.min(steady_y_left), np.min(steady_y_right))
            y_max_in_range = max(np.max(steady_y_left), np.max(steady_y_right))

            y_range_in_box = y_max_in_range - y_min_in_range
            y_padding = y_range_in_box * box_y_padding_factor

            box_y_min = y_min_in_range - y_padding
            box_y_max = y_max_in_range + y_padding
            box_height = box_y_max - box_y_min
            box_defined = True
        else:
            print("Warning: Empty steady range found for y-limits calculation.")
            box_x_start_val, box_width, box_y_min, box_height = 0, 0, 0, 0

    else:
        print(f"Warning: Invalid steady indices provided for bounding box.")
        box_x_start_val, box_width, box_y_min, box_height = 0, 0, 0, 0
else:
     print("Warning: 'result_left' or 'result_right' dictionary or required keys not found. Cannot draw bounding box.")
     box_x_start_val, box_width, box_y_min, box_height = 0, 0, 0, 0


# --- Create Figure and Axes ---
fig, ax_main = plt.subplots(figsize=(8, 5))
ax_twin = ax_main.twinx()

# --- Plotting Data ---
line_load, = ax_main.plot(x_axis, load_data, color='black', linewidth=1, label='Load (N)', zorder=2)
line_left, = ax_twin.plot(x_axis, left_gap, color='blue', linewidth=1.2, label='Left Gap (μm)', zorder=3)
line_right, = ax_twin.plot(x_axis, right_gap, color='red', linewidth=1.2, label='Right Gap (μm)', zorder=3)

# --- Axis Formatting with Dynamic Limits ---
# Left axis (Load)
ax_main.set_xlabel("Axial Distance (mm)", fontsize=12)
ax_main.set_ylabel("Load (N)", color='black', fontsize=12)
ax_main.set_ylim(y_load_lim_min, y_load_lim_max)
ax_main.yaxis.set_major_locator(ticker.MultipleLocator(y_load_tick_interval))
ax_main.tick_params(axis='y', labelcolor='black', labelsize=10)
ax_main.tick_params(axis='x', labelcolor='black', labelsize=10)
ax_main.spines['left'].set_color('black')
ax_main.spines['left'].set_linewidth(1.5)

# Right axis (Gaps)
ax_twin.set_ylabel("Gap (μm)", color='red', fontsize=12)
ax_twin.set_ylim(y_gap_lim_min, y_gap_lim_max)
ax_twin.yaxis.set_major_locator(ticker.MultipleLocator(y_gap_tick_interval))
ax_twin.tick_params(axis='y', labelcolor='red', labelsize=10)
ax_twin.spines['right'].set_color('red')
ax_twin.spines['right'].set_linewidth(1.5)

# X-axis
ax_main.set_xlim(x_lim_min, x_lim_max)
ax_main.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_interval))
ax_main.spines['bottom'].set_color('black')
ax_main.spines['bottom'].set_linewidth(1.5)

# --- Hide Unwanted Spines ---
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)
ax_twin.spines['top'].set_visible(False)
ax_twin.spines['left'].set_visible(False)

# --- Grid lines ---
ax_main.grid(True, which='major', axis='both', linestyle='-', linewidth=0.5, color='darkgrey', zorder=0)

# --- Add Dynamic Bounding Box ---
if box_defined:
    rect = patches.Rectangle((box_x_start_val, box_y_min), # Use dynamic bottom-left corner
                             box_width,                  # Width based on steady indices
                             box_height,                 # Height based on data range + padding
                             linewidth=1, edgecolor='#6389b5', facecolor='#c7d5e8', alpha=0.5,
                             zorder=1) # zorder between grid(0) and lines(2+)
    ax_twin.add_patch(rect) # Add patch to the axis where gap data's Y scale is defined
    print(f"Drawing bounding box: x={box_x_start_val:.1f}, width={box_width:.1f}, y={box_y_min:.1f}, height={box_height:.1f}")

# --- Legend ---
lines = [line_load, line_left, line_right]
labels = [l.get_label() for l in lines]
ax_main.legend(lines, labels, loc='best', fontsize=10, facecolor='white', framealpha=0.9)

# --- Final Layout Adjustment ---
plt.tight_layout()
plt.title(f"{data_file.name}")


# --- Save The Picture ---
# --- Define Output Directory and Filenames ---
output_dir = Path("Plot_Result")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving Figure to: Plot_Result/{data_file.stem}.png")
plt.savefig(f"Plot_Result/{data_file.stem}.png", dpi=300)
plt.show()
