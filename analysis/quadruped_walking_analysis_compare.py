from hydrax.utils.logger import LogReader
from hydrax import ROOT
import matplotlib.pyplot as plt
import math
import numpy as np

reader_base = LogReader(ROOT + "/logs/trot_forward_0.3m_s_fl_foot_hold")
reader_pain_reward = LogReader(ROOT + "/logs/trot_forward_0.3m_s_pain_reward_fl_foot_hold") # simulation_20251005_154505
# Get basic info
# reader.print_info()

# column_names = reader.get_column_names()
column_names = reader_pain_reward.get_cost_column_names()
column_names = ['torso_linear_vel_x_yaw_frame', 'torso_linear_vel_y_yaw_frame', 'step_des_FL_x', 'step_des_FL_y', 'step_des_FL_z', 'foot_pos_FL_x', 'foot_pos_FL_y', 'foot_pos_FL_z', 'touchdown_pos_FL_x', 'touchdown_pos_FL_y']
# column_names = ['torso_linear_vel_x_yaw_frame', 'torso_linear_vel_y_yaw_frame', 'torso_angular_vel_yaw_base', 'torso_height', 'torso_height_des']
# column_names = ['foot_offset_FL_x', 'foot_offset_FL_y']
# column_names = ['torso_height', 'torso_height_des']
# column_names = ['FR_foot_force_z', 'FL_foot_force_z', 'RR_foot_force_z', 'RL_foot_force_z']
column_names = ['torso_linear_vel_x_yaw_frame',
                'step_des_FR_x', 'foot_pos_FR_x',
                'torso_angular_vel_yaw_base', 'foot_pos_FR_z',
                'torso_height_des', 'torso_height', 
                'FR_foot_force_z']

# sanity checks, not guaranteed to be logged every time
# column_names = ["body_weight"]

# print(reader.get_statistics('torso_linear_vel_x_yaw_frame'))
# print(reader.get_statistics('torso_linear_vel_y_yaw_frame'))
# print(reader.get_statistics('torso_angular_vel_yaw_base'))

time_start = 0.0
time_end = 8.0

times_base, data_base = reader_base.get_series_data(column_names)
times_pain_reward, data_pain_reward = reader_pain_reward.get_series_data(column_names)

# Function to clip data based on time range
def clip_data_by_time(times, data_dict, start_time=None, end_time=None):
    if start_time is None and end_time is None:
        return times, data_dict
    
    # Create time mask
    mask = np.ones(len(times), dtype=bool)
    if start_time is not None:
        mask = mask & (times >= start_time)
    if end_time is not None:
        mask = mask & (times <= end_time)
    
    # Apply mask to times
    clipped_times = times[mask]
    
    # Apply mask to all data columns
    clipped_data = {}
    for column, values in data_dict.items():
        clipped_data[column] = values[mask]
    
    return clipped_times, clipped_data

# Clip data to specified time range
times_base_clipped, data_base_clipped = clip_data_by_time(times_base, data_base, time_start, time_end)
times_pain_reward_clipped, data_pain_reward_clipped = clip_data_by_time(times_pain_reward, data_pain_reward, time_start, time_end)

n_columns = len(column_names)
n_cols = min(4, n_columns)  # Maximum 4 columns per row
n_rows = math.ceil(n_columns / n_cols)

# Calculate figure size based on number of subplots
fig_width = 5 * n_cols
fig_height = 4 * n_rows

# Create comparison plots
plt.figure(figsize=(fig_width, fig_height))

# Plot each column in a separate subplot
for i, column in enumerate(column_names):
    plt.subplot(n_rows, n_cols, i + 1)
    
    # Plot both datasets
    plt.plot(times_base_clipped, data_base_clipped[column], label='Base', alpha=0.7)
    plt.plot(times_pain_reward_clipped, data_pain_reward_clipped[column], label='Pain Reward', alpha=0.7)
    
    # # Add threshold line for force columns (optional)
    # if 'force' in column.lower():
    #     threshold = 0.7 * 149.1  # GO2 body weight threshold
    #     plt.axhline(y=threshold, color='red', linestyle='--', 
    #                label=f'Threshold ({threshold:.1f} N)', alpha=0.8)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title(f'{column.replace("_", " ").title()}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.suptitle('Comparison: Base vs Pain Reward')
plt.tight_layout()
plt.show()