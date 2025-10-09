import mujoco
from mujoco import mjx
from typing import Tuple
import numpy as np

from hydrax.task_base import Task

_POST_PHYSICS_STEP_FUNCTIONS = {}

def post_physics_step(
        model: mujoco.MjModel, 
        data: mujoco.MjData,
        task: Task
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Custom function to be called after each physics step to apply further modifications
    to the simulation state
    """
    step_dict = task.post_physics_step()
    available_functions = get_registered_functions()

    for func_name, params in step_dict.items():
        if func_name in available_functions:
            func = available_functions[func_name]
            model, data = func(model, data, **params)
            # print(f"Applied post_physics_step function: {func_name} with params {params}")
        else:
            print(f"Warning: Unrecognized post_physics_step function '{func_name}' - skipping")
            print(f"Available functions:", list(available_functions.keys()))

    return model, data


###########################################################################################################
##### Register custom post_physics_step functions here
###########################################################################################################

def register_post_physics_function(func): 
    """Decorator to register a post_physics_step function."""
    _POST_PHYSICS_STEP_FUNCTIONS[func.__name__] = func
    # print(f"Registered post_physics_step function: {func.__name__}")
    return func

def get_registered_functions():
    """Get all registered post-physics functions."""
    return _POST_PHYSICS_STEP_FUNCTIONS.copy()

@register_post_physics_function
def push_sites(
        model: mujoco.MjModel, 
        data: mujoco.MjData,
        sites: list[str], 
        max_force : list = [1.0, 1.0, 1.0],
        time_interval: float = 0,
        start_time : float = 0,
               ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """ 
        Random pushes the robots. Emulates an impulse by setting a randomized velocity at push site.
        
        Args:
            model: The Mujoco model.
            data: The Mujoco data.
            sites: List of site names to apply pushes.
            max_force: Maximum force to apply in each direction (x, y, z).
            time_interval: Time interval (in simulation steps) between pushes.
                0 means constantly apply
    """
    if data.time < start_time:
        return model, data
    
    if time_interval > 0 and (data.time - start_time) % time_interval > model.opt.timestep: 
        return model, data
    
    for site_name in sites:
        # Get site ID
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        except:
            print(f"Warning: Site '{site_name}' not found")
            continue
            
        # Get the body that the site is attached to
        site_bodyid = model.site_bodyid[site_id]
        
        # Generate random forces (3D: x, y, z)
        random_force = np.random.uniform(-np.array(max_force), np.array(max_force), 3)
        
        # Apply force to the body at the site location
        # xfrc_applied[body_id] = [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        data.xfrc_applied[site_bodyid, :3] += random_force
        # print(f"Applied force {random_force} to site '{site_name}' on body ID {site_bodyid}")
    return model, data

@register_post_physics_function
def push_to_limits(
        model: mujoco.MjModel, 
        data: mujoco.MjData,
        joints: list[str],
        time_interval: float = 0,
        start_time : float = 0,
               ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """ 
        Push the robot to joint limits by setting a random velocity within the max limits.
        Args:
            model: The Mujoco model.
            data: The Mujoco data.
            joints: List of joint names to push to limits.
            time_interval: Time interval (in simulation steps) between pushes, 0 means constantly apply
            start_time: Start time (in seconds) to begin applying pushes.
    """
    if data.time < start_time:
        return model, data
    
    if time_interval > 0 and (data.time - start_time) % time_interval > model.opt.timestep: 
        return model, data
    
    for joint_name in joints:
        # Get joint ID
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        except:
            print(f"Warning: Joint '{joint_name}' not found")
            continue
            
        # Get joint limits
        joint_range = model.jnt_range[joint_id]
        if np.isinf(joint_range).any():
            print(f"Warning: Joint '{joint_name}' has infinite limits, skipping.")
            continue
        
    if np.random.random() < 0.5:
        # Set to lower limit
        data.qpos[joint_id] = joint_range[0]
        # Add velocity to push back toward center (positive direction)
        data.qvel[joint_id] = np.random.uniform(0.1, 0.5)
    else:
        # Set to upper limit  
        data.qpos[joint_id] = joint_range[0] # 1
        # Add velocity to push back toward center (negative direction)
        data.qvel[joint_id] = np.random.uniform(-0.5, -0.1)
        # print(f"Pushed joint '{joint_name}' to position {random_position} with velocity {data.qvel[joint_id]}")
    return model, data
    
@register_post_physics_function
def broken_joints(
        model: mujoco.MjModel, 
        data: mujoco.MjData,
        joints: list[str],
        time_interval: float = 0,
        start_time : float = 0,
               ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """ 
        Simulate a broken joint by setting its position and velocity to zero.
        
        Args:
            model: The Mujoco model.
            data: The Mujoco data.
            joints: List of joint names to break.
            time_interval: Time interval (in simulation steps) between breaking joints, 0 means constantly apply
            start_time: Start time (in seconds) to begin breaking joints.
    """
    if data.time < start_time:
        return model, data
    
    if time_interval > 0 and (data.time - start_time) % time_interval > model.opt.timestep: 
        return model, data
    
    for joint_name in joints:
        # Get joint ID
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        except:
            print(f"Warning: Joint '{joint_name}' not found")
            continue
        
        # Find actuator for this joint
        actuator_id = None
        for i in range(model.nu):
            if model.actuator_trnid[i, 0] == joint_id:
                actuator_id = i
                break
            
        # Set actuator gains to zero (kp=0, kd=0)
        model.actuator_gainprm[actuator_id, 0] = 0.0  # kp (position gain)
        model.actuator_gainprm[actuator_id, 1] = 5.0  # kd (damping gain)
        model.actuator_biasprm[actuator_id, 0] = 0.0  # bias = 0
        model.actuator_biasprm[actuator_id, 1] = 0.0  # bias slope = 0
        
        # print(f"Broke joint '{joint_name}' by setting position and velocity to zero")
    return model, data

@register_post_physics_function
def fix_sites_pos(
        model: mujoco.MjModel, 
        data: mujoco.MjData,
        sites: list[str], 
        start_time : float = 0,
               ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """ 
        Fix the position of specified sites to a target position in world coordinates.
        
        Args:
            model: The Mujoco model.
            data: The Mujoco data.
            sites: List of site names to fix.
            start_time: Start time (in seconds) to begin fixing sites.
    """
    if data.time < start_time:
        return model, data
    
    if not hasattr(fix_sites_pos, "first_execution"):
        fix_sites_pos.first_execution = True
        fix_sites_pos.target_positions = {}
        for site_name in sites:
            try: 
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                fix_sites_pos.target_positions[site_name] = data.site_xpos[site_id].copy()
            except:
                continue
    
    for site_name in sites:
        # Get site ID
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        except:
            print(f"Warning: Site '{site_name}' not found")
            continue
            
        # Get the body that the site is attached to
        site_bodyid = model.site_bodyid[site_id]
        
        # Current position of the site in world coordinates
        target_pos = fix_sites_pos.target_positions.get(site_name, None)
        current_pos = data.site_xpos[site_id]
        
        # Compute position error
        pos_error = np.array(target_pos) - current_pos
        
        # Apply corrective force proportional to the position error
        kp = 5000.0  # Position gain
        corrective_force = kp * pos_error
        kd = 500.0
        damping_force = -kd * data.cvel[site_bodyid][3:6]
        corrective_force += damping_force
        # Apply force to the body at the site location
        data.xfrc_applied[site_bodyid, :3] += corrective_force
        # print(f"Applied corrective force {corrective_force} to fix site '{site_name}' at position {target_pos}")
    return model, data
