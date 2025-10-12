from hydrax.utils.logger import LogReader
from hydrax import ROOT

reader = LogReader(ROOT + "/logs/simulation_20251012_182626")

# Get basic info
reader.print_info()

column_names = reader.get_column_names()
column_names = reader.get_cost_column_names()
# column_names = ['torso_linear_vel_x_yaw_frame', 'torso_linear_vel_y_yaw_frame', 'step_des_FL_x', 'step_des_FL_y', 'step_des_FL_z', 'foot_pos_FL_x', 'foot_pos_FL_y', 'foot_pos_FL_z', 'touchdown_pos_FL_x', 'touchdown_pos_FL_y']
column_names = ['step_des_FL_x', 'step_des_FR_x', 'step_des_FL_z', 'step_des_FR_z', 
                'foot_pos_FL_x', 'foot_pos_FR_x', 'foot_pos_FL_z', 'foot_pos_FR_z',]
# column_names = ["FL_hip_joint", "FL_hip_control"]
# column_names = ['torso_linear_vel_x_yaw_frame', 'torso_linear_vel_y_yaw_frame', 'torso_angular_vel_yaw_base', 'torso_height', 'torso_height_des']
# column_names = ['foot_offset_FL_x', 'foot_offset_FL_y']
# column_names = ['torso_height', 'torso_height_des']
# column_names = ['FR_foot_force_z', 'FL_foot_force_z', 'RR_foot_force_z', 'RL_foot_force_z']
# column_names = ['base_force_x', 'base_force_y', 'base_force_z',]
# column_names = ["FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]

# sanity checks, not guaranteed to be logged every time
# column_names = ["body_weight"]

print(reader.get_statistics('torso_linear_vel_x_yaw_frame'))
# print(reader.get_statistics('torso_linear_vel_y_yaw_frame'))
# print(reader.get_statistics('torso_angular_vel_yaw_base'))
reader.plot_time_series(column_names)