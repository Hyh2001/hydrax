from hydrax.utils.logger import LogReader
from hydrax import ROOT

reader = LogReader(ROOT + "/logs/simulation_20250926_164813")

# Get basic info
reader.print_info()

# column_names = reader.get_column_names()
column_names = reader.get_cost_column_names()
# column_names = ['torso_linear_vel_x_yaw_frame', 'torso_linear_vel_y_yaw_frame', 'step_des_FL_x', 'step_des_FL_y', 'step_des_FL_z', 'foot_pos_FL_x', 'foot_pos_FL_y', 'foot_pos_FL_z', 'td_pos_FL_x', 'td_pos_FL_y']
# column_names = ['foot_offset_FL_x', 'foot_offset_FL_y']

reader.plot_time_series(column_names)