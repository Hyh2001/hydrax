from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

from mujoco.mjx._src.math import quat_inv

class QuadrupedStandup(Task):
    """Standup task for the Go2."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/go2/flat_ground.xml")
        super().__init__(
            mj_model,
            trace_sites=["imu", "FR_grf_sensor", "FL_grf_sensor", "RL_grf_sensor", "RR_grf_sensor"],
        )

        # Get sensor and site ids
        self.orientation_sensor_id = mj_model.sensor("imu_quat").id
        self.velocity_sensor_id = mj_model.sensor("frame_vel_global").id
        self.torso_id = mj_model.site("imu").id
        # Get foot site IDs
        self.fr_foot_id = self.mj_model.site("FR_grf_sensor").id
        self.fl_foot_id = self.mj_model.site("FL_grf_sensor").id 
        self.rl_foot_id = self.mj_model.site("RL_grf_sensor").id
        self.rr_foot_id = self.mj_model.site("RR_grf_sensor").id

        # Set the target height
        self.target_height = 0.25
        self.target_linear_velocity = jnp.array([0.0, 0.0])  # m/s
        self.target_angular_velocity = jnp.array([0.0])  # rad/s

        # Standing configuration
        self.qstand = jnp.array(mj_model.keyframe("stand").qpos)

    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get the height of the torso above the ground."""
        return state.site_xpos[self.torso_id, 2]

    def _get_torso_orientation(self, state: mjx.Data) -> jax.Array:
        """Get the rotation from the current torso orientation to upright."""
        sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
        quat = state.sensordata[sensor_adr : sensor_adr + 4]
        upright = jnp.array([0.0, 0.0, 1.0])
        return mjx._src.math.rotate(upright, quat_inv(quat))
    
    def _get_foot_positions_intersection(self, state: mjx.Data) -> jax.Array:
        """Get the intersection point of lines formed by diagonal foot pairs (FR-RL and FL-RR)."""
        # Get foot positions (only x, y coordinates)
        fr_pos = state.site_xpos[self.fr_foot_id, :2]  # Front Right
        fl_pos = state.site_xpos[self.fl_foot_id, :2]  # Front Left
        rl_pos = state.site_xpos[self.rl_foot_id, :2]  # Rear Left
        rr_pos = state.site_xpos[self.rr_foot_id, :2]  # Rear Right
        
        # Simple geometric center (centroid of the quadrilateral)
        support_center = (fr_pos + fl_pos + rl_pos + rr_pos) / 4.0
        
        return support_center

    def _get_foot_positions_z(self, state: mjx.Data) -> jax.Array:
        """Get the z-coordinates of all four feet."""
        fr_z = state.site_xpos[self.fr_foot_id, 2]  # Front Right
        fl_z = state.site_xpos[self.fl_foot_id, 2]  # Front Left
        rl_z = state.site_xpos[self.rl_foot_id, 2]  # Rear Left
        rr_z = state.site_xpos[self.rr_foot_id, 2]  # Rear Right
        
        return jnp.array([fr_z, fl_z, rl_z, rr_z])

    def _quat_to_yaw(self, quat: jax.Array) -> jax.Array:
        """Convert a quaternion to a yaw angle."""
        # Quaternion components
        w, x, y, z = quat
        # Yaw calculation
        yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def _get_torso_yaw(self, state: mjx.Data) -> jax.Array:
        """Get the yaw angle of the torso."""
        sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
        quat = state.sensordata[sensor_adr : sensor_adr + 4]
        return self._quat_to_yaw(quat)
    
    def _get_torso_linear_velocity_xy(self, state: mjx.Data) -> jax.Array:
        """Get the xy linear velocity of the torso"""
        sensor_adr = self.model.sensor_adr[self.velocity_sensor_id]
        vel = state.sensordata[sensor_adr : sensor_adr + 3]
        return vel[:2]  # xy-direction velocity

    def _get_torso_angular_velocity_yaw(self, state: mjx.Data) -> jax.Array:
        """Get the yaw angular velocity of the torso"""
        sensor_adr = self.model.sensor_adr[self.velocity_sensor_id]
        vel = state.sensordata[sensor_adr + 3 : sensor_adr + 6]
        return vel[2]  # yaw angular velocity

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        orientation_cost = jnp.sum(
            jnp.square(self._get_torso_orientation(state)[0:2])
        )
        # jax.debug.print("projected gravity {}", self._get_torso_orientation(state))
        height_cost = jnp.square(
            self._get_torso_height(state) - self.target_height
        )
        pos_cost = jnp.sum(jnp.square(state.qpos[0:2] - self.qstand[0:2]))
        # com_cost = jnp.sum(jnp.square(self._get_foot_positions_intersection(state) - state.qpos[0:2]))
        foot_pos_cost = jnp.sum(jnp.square(self._get_foot_positions_z(state)))  # Penalize feet being off the ground (z > 0)
        
        # Anti-jittering: penalize deviation from standing control
        u_ref = self.qstand[7:]  # Use standing joint angles as control reference
        control_smoothness_cost = jnp.sum(jnp.square(control - u_ref))
        control_cost = jnp.sum(jnp.square(control))
        linear_velocity_cost = jnp.sum(jnp.square(self._get_torso_linear_velocity_xy(state) - self.target_linear_velocity))  # target forward velocity 0.5 m/s
        angular_velocity_cost = jnp.sum(jnp.square(self._get_torso_angular_velocity_yaw(state) - self.target_angular_velocity))  # target yaw angular velocity 0 rad/s
        yaw_cost = jnp.square(self._get_torso_yaw(state))

        return (10.0 * orientation_cost + 
                10.0 * height_cost + 
                10.0 * pos_cost + 
                1.0 * control_smoothness_cost) #works for position control
        # return (0.5 * orientation_cost + 
        #         0.3 * yaw_cost +
        #         1.0 * height_cost + 
        #         0.1 * foot_pos_cost + 
        #         1.0 * linear_velocity_cost +
        #         1.0 * angular_velocity_cost +
        #         0 * control_cost) # + 0.5 * control_smoothness_cost# 40 for orientation with + 0.5 * control_smoothness_cost


    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return 1.2*self.running_cost(state, jnp.zeros(self.model.nu)) # 1.2 for position control

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

    def log_data(self):
        return super().log_costs()