import argparse

import mujoco
import numpy as np
import jax.numpy as jnp

from hydrax.algs import MPPI, DIAL
from hydrax.simulation.asynchronous import run_interactive as run_async
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.quadruped_walking import QuadrupedWalking

"""
Run an interactive simulation of the quadrupedal standup task.
"""

# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of quaduped (GO2) standup."
    )
    parser.add_argument(
        "-a",
        "--asynchronous",
        action="store_true",
        help="Use asynchronous simulation",
        default=False,
    )
    args = parser.parse_args()

    # Define the task (cost and dynamics)
    task = QuadrupedWalking()

    # roughly works
    # ctrl = MPPI(
    #     task,
    #     num_samples=2048,       
    #     noise_level=0.07,     
    #     temperature=0.01,     
    #     num_randomizations=1, 
    #     plan_horizon=0.4,     
    #     spline_type="zero",
    #     num_knots=4,         
    # )
    # position
    ctrl = MPPI(
        task,
        num_samples=2048,       
        noise_level=0.1,  # 
        temperature=0.5,  # 0.5 proportional to cost level, cost gain 1 -> temperature 0.1 
        num_randomizations=1, 
        plan_horizon=0.6,     
        spline_type="zero",  # zero
        num_knots=4,         
    )
    # torque
    # ctrl = MPPI(
    #     task,
    #     num_samples=2048,       
    #     noise_level=1.0,  # 0.4, 0.1, 0.03
    #     temperature=10,  # 2.0, 1.0, 1.0, 0.07 proportional to cost level, cost gain 1 -> temperature 0.1 
    #     num_randomizations=1, 
    #     plan_horizon=0.6,     
    #     spline_type="zero",
    #     num_knots=4,         
    # )
    
    
    # DIAL MPC original
    # ctrl = DIAL(
    #     task,
    #     num_samples=2048,
    #     noise_level=1.0, # 1.0 for original 
    #     beta_opt_iter=0.5, # 0.01, 0.5 for original
    #     beta_horizon=0.9, # 1, 0.9 for original
    #     temperature=0.06, # 0.1, 0.06 for original
    #     plan_horizon=0.4,
    #     spline_type="zero",
    #     num_knots=4,
    #     iterations=2,
    # )
    
    
    # Define the model used for simulation - OPTIMIZED FOR REALTIME
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.01   # 0.01   
    mj_model.opt.iterations = 10        
    mj_model.opt.ls_iterations = 50   
    mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
    # mj_model.opt.o_solimp = [0.8, 0.8, 0.01, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

    # Set the initial state so the robot falls and needs to stand back up
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = mj_model.keyframe("stand").qpos
    # mj_data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0] 
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] 
    # mj_data.userdata = np.zeros(16)
    initial_knots = jnp.tile(task.qstand[7:], (ctrl.num_knots, 1))
    
    # Run the interactive simulation
    if args.asynchronous:
        print("Running asynchronous simulation")
        mj_model.opt.timestep = 0.01
        mj_model.opt.iterations = 100
        mj_model.opt.ls_iterations = 50
        mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

        run_async(
            ctrl,
            mj_model,
            mj_data,
        )
    else:
        print("Running deterministic simulation")
        run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=50,
            initial_knots=initial_knots,      
            show_traces=False,
            enable_logging=True,
            record_video=True,
        )
