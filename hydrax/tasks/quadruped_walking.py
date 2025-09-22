from typing import Dict, Callable
from datetime import datetime

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src.math import quat_inv

from hydrax import ROOT
from hydrax.task_base import Task

# from hydrax.utils.logger import JAXLogger


class QuadrupedWalking(Task):
    """Waling task for the quadrupedal robots."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/go2/flat_ground.xml")
        super().__init__(
            mj_model,
            trace_sites=["imu", 
                         "FL_grf_sensor", "FR_grf_sensor", "RL_grf_sensor", "RR_grf_sensor", 
                         "FL_hip_site", "FR_hip_site", "RL_hip_site", "RR_hip_site"],
        )

        # Get sensor and site ids 
        # all in the world frame
        self.orientation_sensor_id = mj_model.sensor("imu_quat").id
        self.velocity_sensor_id = mj_model.sensor("frame_vel_global").id
        self.angular_velocity_id = mj_model.sensor("frame_angvel_global").id
        self.torso_id = mj_model.site("imu").id
        # print(f"velocity_sensor_id: {self.velocity_sensor_id}")
        # print(f"torso_id: {self.torso_id}")
        
        # Set the target height
        self.target_height = 0.27
        
        # Standing configuration
        self.qstand = jnp.array(mj_model.keyframe("stand").qpos)
        
        # gait
        self.gait = "trot"
        self._gait_phase = {
            "stand": jnp.zeros(4),
            "walk": jnp.array([0.0, 0.5, 0.75, 0.25]),
            "trot": jnp.array([0.0, 0.5, 0.5, 0.0]),
            "canter": jnp.array([0.0, 0.33, 0.33, 0.66]),
            "gallop": jnp.array([0.0, 0.05, 0.4, 0.35]),
        }
        self._gait_params = {
            # duty_ratio, cadence, amplitude
            "stand": jnp.array([1.0, 1.0, 0.0]),
            "walk": jnp.array([0.75, 1.0, 0.08]),
            # "trot": jnp.array([0.45, 2.0, 0.08]),
            "trot": jnp.array([0.45, 2.0, 0.03]),
            "canter": jnp.array([0.4, 4.0, 0.06]),
            "gallop": jnp.array([0.3, 3.5, 0.10]),
        }
        self._feet_site_id = jnp.array([
            mj_model.site("FL_grf_sensor").id,
            mj_model.site("FR_grf_sensor").id, 
            mj_model.site("RL_grf_sensor").id,
            mj_model.site("RR_grf_sensor").id,
        ])
        self._hip_site_id = jnp.array([
            mj_model.site("FL_hip_site").id,
            mj_model.site("FR_hip_site").id, 
            mj_model.site("RL_hip_site").id,
            mj_model.site("RR_hip_site").id,
        ])
        # self.target_linear_velocity = jnp.array([0.0, 0.0])  # m/s in the local frame
        self.target_linear_velocity = jnp.array([0.0, 0.0])  # m/s in the local frame
        self.target_angular_velocity = jnp.array([0.0])  # rad/s in the local frame

        # get the foot offset from hip from qpos
        self.foot_offset_xy = self._calculate_foot_offset_xy()
        # self.touchdown_positions = jnp.zeros((4,3))
        # self.last_phases = jnp.zeros(4) #0.02  # m, foot radius
        
    def _calculate_foot_offset_xy(self) -> jax.Array:
        # Create a temporary data structure with standing configuration
        temp_data = mjx.make_data(self.model)
        temp_data = temp_data.replace(qpos=self.qstand)
        
        # Forward kinematics to get positions
        temp_data = mjx.forward(self.model, temp_data)
        
        # Get hip and foot positions in standing configuration
        hip_pos_standing = temp_data.site_xpos[self._hip_site_id, :3]  # (4, 3)
        foot_pos_standing = temp_data.site_xpos[self._feet_site_id, :3]  # (4, 3)
        
        # Calculate foot offset from hip in world frame
        hip_to_foot_world = foot_pos_standing - hip_pos_standing  # (4, 3)
        
        # Since this is calculated from standing pose (yaw=0), these are already in "yaw-aligned" frame
        return hip_to_foot_world[:, :2]  # (4, 2)
        
    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get the height of the torso above the ground."""
        return state.site_xpos[self.torso_id, 2]
    
    def _get_torso_position(self, state: mjx.Data) -> jax.Array:
        """Get the x, y, z coordinates of the torso."""
        return state.site_xpos[self.torso_id, :3] # x, y, z coordinates of torso (1,3)

    def _get_torso_orientation(self, state: mjx.Data) -> jax.Array:
        """Get the rotation from the current torso orientation to upright."""
        sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
        quat = state.sensordata[sensor_adr : sensor_adr + 4]
        upright = jnp.array([0.0, 0.0, 1.0])
        return mjx._src.math.rotate(upright, quat_inv(quat))
    
    def _get_torso_yaw(self, state: mjx.Data) -> jax.Array:
        """Get the yaw angle of the torso"""
        sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
        quat = state.sensordata[sensor_adr : sensor_adr + 4]
        
        # Convert quaternion to yaw angle
        # quat = [w, x, y, z] in MuJoCo format
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Calculate yaw angle using atan2
        # Formula: yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        return yaw
    
    def _get_torso_linear_velocity(self, state: mjx.Data) -> jax.Array:
        """Get the 3D linear velocity of the torso"""
        sensor_adr = self.model.sensor_adr[self.velocity_sensor_id]
        vel = state.sensordata[sensor_adr : sensor_adr + 3]
        return vel  # 3D linear velocity
    
    def _get_torso_angular_velocity(self, state: mjx.Data) -> jax.Array:
        """Get the 3D angular velocity of the torso"""
        sensor_adr = self.model.sensor_adr[self.angular_velocity_id]
        vel = state.sensordata[sensor_adr: sensor_adr + 3]
        return vel  # 3D angular velocity
    
    def _get_torso_linear_velocity_xy_base(self, state: mjx.Data) -> jax.Array:
        """Get the xy linear velocity of the torso"""
        sensor_adr = self.model.sensor_adr[self.velocity_sensor_id]
        vel = state.sensordata[sensor_adr : sensor_adr + 3]
        vel = mjx._src.math.rotate(vel, mjx._src.math.quat_inv(state.xquat[self.torso_id]))  # to the base frame
        return vel[:2]  # xy-direction velocity in base frame

    def _get_torso_linear_velocity_xy_yaw_base(self, state: mjx.Data) -> jax.Array:
        """Get the xy linear velocity of the torso in yaw-aligned base frame"""
        sensor_adr = self.model.sensor_adr[self.velocity_sensor_id]
        vel = state.sensordata[sensor_adr : sensor_adr + 3]
        
        # Get current yaw
        current_yaw = self._get_torso_yaw(state)
        yaw_quat = jnp.array([
            jnp.cos(current_yaw / 2),  # w
            0.0,                       # x 
            0.0,                       # y
            jnp.sin(current_yaw / 2)   # z
        ])
        yaw_quat_inv = mjx._src.math.quat_inv(yaw_quat)
        
        # Rotate to yaw-aligned base frame
        vel = mjx._src.math.rotate(vel, yaw_quat_inv)  # to the yaw-aligned base frame
        return vel[:2]  # xy-direction velocity in yaw-aligned base frame

    def _get_torso_angular_velocity_yaw_base(self, state: mjx.Data) -> jax.Array:
        """Get the yaw angular velocity of the torso"""
        vel = self._get_torso_angular_velocity(state)
        vel = mjx._src.math.rotate(vel, mjx._src.math.quat_inv(state.xquat[self.torso_id]))  # to the base frame
        return vel[2]
        # return jnp.clip(vel[2], -2.0, 2.0)  # yaw angular velocity in base frame
    
    def _get_foot_positions(self, state: mjx.Data) -> jax.Array:
        """Get the x, y, z coordinates of all four feet."""
        feet_pos = state.site_xpos[self._feet_site_id, :3]  # x, y, z coordinates of feet
        return feet_pos
    
    def _get_hip_positions(self, state: mjx.Data) -> jax.Array:
        """Get the x, y, z coordinates of all four hips."""
        hip_pos = state.site_xpos[self._hip_site_id, :3]  # x, y, z coordinates of hips
        return hip_pos
    
    def _raibert_heuristic(self, state: mjx.Data) -> jax.Array:
        """Raibert heuristic to determine desired foot positions based on torso position and yaw.
        This now properly interpolates between stance and swing phases.
        
        Args:
            state: Current simulation state
        Returns:
            Current desired foot positions (4, 3) interpolated based on gait phase.
        """
        # Get gait parameters
        duty_ratio, cadence, amplitude = self._gait_params[self.gait]
        phases = self._gait_phase[self.gait]
        current_phase = (state.time * cadence + phases) % 1.0  # (4,)
        # is_swing = current_phase > duty_ratio  # (4,)
        
        torso_pos_world = self._get_torso_position(state)  # (3,)
        torso_vel_world = self._get_torso_linear_velocity(state)  # (3,)
        hip_pos_world = self._get_hip_positions(state)  # (4, 3)
        current_yaw = self._get_torso_yaw(state)
        
        yaw_quat = jnp.array([
        jnp.cos(current_yaw / 2),  # w
        0.0,                       # x 
        0.0,                       # y
        jnp.sin(current_yaw / 2)   # z
        ])
        yaw_quat_inv = mjx._src.math.quat_inv(yaw_quat)
        
        # Transform to yaw-aligned base frame
        hip_pos_yaw_frame = jax.vmap(mjx._src.math.rotate, in_axes=(0, None))(
            hip_pos_world - torso_pos_world, yaw_quat_inv
        )  # (4, 3)
        torso_vel_yaw_frame = mjx._src.math.rotate(torso_vel_world, yaw_quat_inv) 
            

        # Calculate stepping time (swing time)
        stepping_time = (1.0 - duty_ratio) / cadence  # Swing time
        
        # Compute desired touchdown positions using Raibert heuristic
        velocity_offset = torso_vel_yaw_frame[:2] * stepping_time / 2.0  # (2,)
        feedback = 0.1 * (torso_vel_yaw_frame[:2] - self.target_linear_velocity) # (2,)
        touchdown_pos_yaw_xy = (hip_pos_yaw_frame[:, :2] + 
                                velocity_offset[None, :] + 
                                feedback[None, :] + 
                                self.foot_offset_xy)# (4, 2)
        
        # compute touchdown pos in world frame
        touchdown_z_yaw_frame = -torso_pos_world[2]
        touchdown_pos_yaw_frame = jnp.concatenate([
            touchdown_pos_yaw_xy,
            jnp.full((4, 1), touchdown_z_yaw_frame)
        ], axis=1)  # (4, 3)
        touchdown_pos_world = jax.vmap(mjx._src.math.rotate, in_axes=(0, None))(
            touchdown_pos_yaw_frame, yaw_quat
        ) + torso_pos_world[None, :]  # (4, 3)
        
        # Get current foot positions
        current_foot_pos = self._get_foot_positions(state)  # (4, 3)
        # Use the unified trajectory calculation
        interpolated_positions = self._compute_foot_trajectory(
            state, current_foot_pos, touchdown_pos_world, 
            duty_ratio, cadence, amplitude, phases
        )
        
        return interpolated_positions
    
    def _get_foot_step(self, duty_ratio, cadence, amplitude, phases, time):
        """Compute the foot step height (kept for compatibility).
        
        This function is now mainly used for getting step heights.
        The main trajectory interpolation is handled by _interpolate_foot_trajectory.
        """
        def step_height(t, footphase, duty_ratio):
            angle = (t + jnp.pi - footphase) % (2 * jnp.pi) - jnp.pi
            angle = jnp.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
            clipped_angle = jnp.clip(angle, -jnp.pi / 2, jnp.pi / 2)
            value = jnp.where(duty_ratio < 1, jnp.cos(clipped_angle), 0)
            final_value = jnp.where(jnp.abs(value) >= 1e-6, jnp.abs(value), 0.0)
            return final_value
        
        h_steps = amplitude * jax.vmap(step_height, in_axes=(None, 0, None))(
            time * 2 * jnp.pi * cadence + jnp.pi,
            2 * jnp.pi * phases,
            duty_ratio,
        )
        return h_steps
    
    def _compute_foot_trajectory(self, 
                                 state: mjx.Data,
                                 current_pos: jax.Array,
                                 touchdown_pos: jax.Array,
                                 duty_ratio: float,
                                 cadence: float,
                                 amplitude: float,
                                 phases: jax.Array) -> jax.Array:
        """Compute complete foot trajectory combining gait phase, step height, and interpolation.
    
        Args:
            state: Current simulation state (for time)
            current_pos: Current foot positions (4, 3)
            touchdown_pos: Desired touchdown positions (4, 3)
            duty_ratio: Fraction of time foot is on ground
            cadence: Gait cadence (steps per second)
            amplitude: Maximum step height during swing phase
            phases: Phase offsets for each foot (4,)
            
        Returns:
            Complete foot trajectory positions (4, 3)
        """
        def single_foot_complete_trajectory(current, touchdown, phase_offset, duty_ratio, cadence, amplitude, time):
            """Compute complete trajectory for a single foot with integrated step height calculation."""
            # calculate current phase
            current_phase = (time * cadence + phase_offset) % 1.0
            # Determine if foot is in stance or swing phase
            is_swing = current_phase > duty_ratio
            # Calculate swing progress (0 to 1 during swing phase)
            swing_progress = jnp.where(
                is_swing,
                (current_phase - duty_ratio) / (1.0 - duty_ratio),
                0.0
            )
            # design step height profile
            angle = (time * 2 * jnp.pi * cadence + jnp.pi + jnp.pi - phase_offset * 2 * jnp.pi) % (2 * jnp.pi) - jnp.pi
            angle = jnp.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
            clipped_angle = jnp.clip(angle, -jnp.pi / 2, jnp.pi / 2)
            step_value = jnp.where(duty_ratio < 1, jnp.cos(clipped_angle), 0)
            step_height_normalized = jnp.where(jnp.abs(step_value) >= 1e-6, jnp.abs(step_value), 0.0)
            step_height = amplitude * step_height_normalized + 0.022 # add foot geometry radius
            
            # Interpolation
            final_pos = jnp.where(
                is_swing,
                # During swing: interpolate XY from current to touchdown, add step height to Z
                jnp.array([
                    current[0] + swing_progress * (touchdown[0] - current[0]),  # X interpolation
                    current[1] + swing_progress * (touchdown[1] - current[1]),  # Y interpolation
                    touchdown[2] + step_height  # Z = ground level + step height
                ]),
                # During stance: stay at current position (don't move!)
                current
            )
                
            return final_pos
        
        # Apply to all feet
        trajectory = jax.vmap(
            single_foot_complete_trajectory, 
            in_axes=(0, 0, 0, None, None, None, None)
        )(current_pos, touchdown_pos, phases, duty_ratio, cadence, amplitude, state.time)
        
        return trajectory
    
    def _get_yaw_cost(self, state: mjx.Data) -> jax.Array:
        """Compute yaw tracking cost with time-varying target."""
        
        # Get current yaw
        current_yaw = self._get_torso_yaw(state)
        
        # Calculate time-varying target yaw
        # You'd need to track these in your task state
        yaw_tar = (
            0.0
            + self.target_angular_velocity[2] * state.time
        )
        
        # Compute wrapped yaw error
        d_yaw = current_yaw - yaw_tar
        wrapped_error = jnp.arctan2(jnp.sin(d_yaw), jnp.cos(d_yaw))
        
        return jnp.square(wrapped_error)
    
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        orientation_cost = jnp.sum(
            jnp.square(self._get_torso_orientation(state)[0:2])
        )
        height_cost = jnp.square(
            self._get_torso_height(state) - self.target_height
        )

        # linear_velocity_cost = jnp.sum(jnp.square(self._get_torso_linear_velocity_xy_base(state) - self.target_linear_velocity))  # in base frame
        linear_velocity_cost = jnp.sum(jnp.square(self._get_torso_linear_velocity_xy_yaw_base(state) - self.target_linear_velocity))  # in yaw-aligned base frame
        z_linear_velocity_cost = jnp.square(self._get_torso_linear_velocity(state)[2])  # target vertical velocity 0 m/s
        # angular_velocity_cost = jnp.sum(jnp.square(self._get_torso_angular_velocity_yaw_base(state) - self.target_angular_velocity))  # in base frame
        angular_velocity_cost = jnp.sum(jnp.square(self._get_torso_angular_velocity(state)[2] - self.target_angular_velocity))  # in world frame
        yaw_cost = self._get_yaw_cost(state)
        
        # Gait cost (foot trajectory tracking)
        feet_target = self._raibert_heuristic(state)  # Desired foot positions (4, 3)
        feet_error = feet_target - self._get_foot_positions(state)  # (4, 3)
        xy_error = feet_error[:, :2]/0.05
        z_error = feet_error[:, 2]/0.05
        gait_cost = jnp.sum(xy_error**2) + 100*jnp.sum(z_error**2)
        # gait_cost = jnp.sum(((feet_target - self._get_foot_positions(state)) / 0.05) ** 2)
        
        # Gait cost (only z positions)
        # duty_ratio, cadence, amplitude = self._gait_params[self.gait]
        # phases = self._gait_phase[self.gait]
        # feet_target = self._get_foot_step(duty_ratio, cadence, amplitude, phases, state.time)
        # # we should take into account the foot radius (0.022 m)
        # gait_cost = jnp.sum(((feet_target + 0.022 - self._get_foot_positions(state)[:,2]) / 0.05) ** 2)
         
        return 10*(8 * orientation_cost + 
                200 * height_cost + 
                20 * yaw_cost +
                10.0 * linear_velocity_cost + 
                # 10.0 * z_linear_velocity_cost +
                10.0 * angular_velocity_cost + # should not be too big
                1.0 * gait_cost # 0.3 for z-only, 0.5 roughly walks two step, 1.0 for raibert
                )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return 1.0*self.running_cost(state, jnp.zeros(self.model.nu)) 

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the friction parameters."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly perturb the measured base position and velocities."""
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.01 * jax.random.normal(q_rng, (7,))
        v_err = 0.01 * jax.random.normal(v_rng, (6,))

        qpos = data.qpos.at[0:7].set(data.qpos[0:7] + q_err)
        qvel = data.qvel.at[0:6].set(data.qvel[0:6] + v_err)

        return {"qpos": qpos, "qvel": qvel}
    
    def log_data(self, state: mjx.Data, control: jax.Array) -> Dict[str, jax.Array]:
        data_dict = {}
        data_dict["total_cost"] = self.running_cost(state, control)
        data_dict["orientation_cost"] = jnp.sum(
            jnp.square(self._get_torso_orientation(state)[0:2])
        )
        data_dict["height_cost"] = jnp.square(
            self._get_torso_height(state) - self.target_height
        )
        data_dict["linear_velocity_cost"] = jnp.sum(jnp.square(self._get_torso_linear_velocity_xy_base(state) - self.target_linear_velocity))
        data_dict["z_linear_velocity_cost"] = jnp.square(self._get_torso_linear_velocity(state)[2])
        data_dict["angular_velocity_cost"] = jnp.sum(jnp.square(self._get_torso_angular_velocity(state)[2] - self.target_angular_velocity))
        feet_target = self._raibert_heuristic(state)  # Desired foot positions (4, 3)
        feet_error = feet_target - self._get_foot_positions(state)  # (4, 3)
        xy_error = feet_error[:, :2]/0.05
        z_error = feet_error[:, 2]/0.05
        gait_cost = jnp.sum(xy_error**2) + 10*jnp.sum(z_error**2)
        data_dict["gait_cost"] = gait_cost
        # data_dict["gait_cost"] = jnp.sum(((self._get_foot_step(*self._gait_params[self.gait], self._gait_phase[self.gait], state.time) + 0.022 - self._get_foot_positions(state)[:,2]) / 0.05) ** 2)
        data_dict["yaw_cost"] = self._get_yaw_cost(state)
        
        data_dict["torso_linear_vel_x_base"] = self._get_torso_linear_velocity_xy_base(state)[0]
        data_dict["torso_linear_vel_y_base"] = self._get_torso_linear_velocity_xy_base(state)[1]
        data_dict["torso_angular_vel_yaw_base"] = self._get_torso_angular_velocity_yaw_base(state)
        
        data_dict["torso_height"] = self._get_torso_height(state)
        data_dict["torso_height_des"] = self.target_height
        
        # data_dict["foot_pos_FL_z"] = self._get_foot_positions(state)[0,2]
        data_dict["foot_pos_FR_x"] = self._get_foot_positions(state)[1,0]
        data_dict["foot_pos_FR_y"] = self._get_foot_positions(state)[1,1]
        data_dict["foot_pos_FR_z"] = self._get_foot_positions(state)[1,2]
        # data_dict["foot_pos_RL_z"] = self._get_foot_positions(state)[2,2]
        # data_dict["foot_pos_RR_z"] = self._get_foot_positions(state)[3,2]
        # data_dict["sim_time"] = state.time
        # data_dict["control_cost"] = jnp.sum(jnp.square(control))
        step_des = self._raibert_heuristic(state)  # Desired foot positions (4, 3)

        # data_dict["step_des_FL_x"] = step_des[0,0]
        # data_dict["step_des_FL_y"] = step_des[0,1]
        # data_dict["step_des_FL_z"] = step_des[0,2]
        data_dict["step_des_FR_x"] = step_des[1,0]
        data_dict["step_des_FR_y"] = step_des[1,1]
        data_dict["step_des_FR_z"] = step_des[1,2]
        # data_dict["step_des_FR_z"] = self._get_foot_step(*self._gait_params[self.gait], self._gait_phase[self.gait], state.time)[1]
        # data_dict["step_des_RL_x"] = step_des[2,0]
        # data_dict["step_des_RL_y"] = step_des[2,1]
        # data_dict["step_des_RL_z"] = step_des[2,2]
        # data_dict["step_des_RR_x"] = step_des[3,0]
        # data_dict["step_des_RR_y"] = step_des[3,1]
        # data_dict["step_des_RR_z"] = step_des[3,2]  
        
        return data_dict
        