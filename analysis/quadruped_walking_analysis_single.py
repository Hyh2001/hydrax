from hydrax.utils.logger import LogReader
from hydrax import ROOT

reader = LogReader(ROOT + "/logs/simulation_20251013_165230")

# Get basic info
reader.print_info()

column_names = reader.get_column_names()
column_names = reader.get_cost_column_names()
# column_names = ['torso_linear_vel_x_yaw_frame', 'torso_linear_vel_y_yaw_frame', 'step_des_FL_x', 'step_des_FL_y', 'step_des_FL_z', 'foot_pos_FL_x', 'foot_pos_FL_y', 'foot_pos_FL_z', 'touchdown_pos_FL_x', 'touchdown_pos_FL_y']
# column_names = ['torso_linear_vel_x_yaw_frame', 'torso_linear_vel_y_yaw_frame', 'torso_angular_vel_yaw_base', 'torso_height', 'torso_height_des']
# column_names = ['foot_offset_FL_x', 'foot_offset_FL_y']
# column_names = ['torso_height', 'torso_height_des']
# column_names = ['FR_foot_force_z', 'FL_foot_force_z', 'RR_foot_force_z', 'RL_foot_force_z']
column_names = ["FL_hip_joint", "FL_hip_joint_des", 
                "FL_thigh_joint", "FL_thigh_joint_des", 
                "FL_calf_joint", "FL_calf_joint_des",
                "RL_hip_joint", "RL_hip_joint_des", 
                "RL_thigh_joint", "RL_thigh_joint_des", 
                "RL_calf_joint", "RL_calf_joint_des",
                ]

# sanity checks, not guaranteed to be logged every time
# column_names = ["body_weight"]

print(reader.get_statistics('torso_linear_vel_x_yaw_frame'))
# print(reader.get_statistics('torso_linear_vel_y_yaw_frame'))
# print(reader.get_statistics('torso_angular_vel_yaw_base'))
reader.plot_time_series(column_names)