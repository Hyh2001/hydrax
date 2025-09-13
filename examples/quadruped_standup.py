import argparse

import mujoco

from hydrax.algs import MPPI, DIAL
from hydrax.simulation.asynchronous import run_interactive as run_async
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.quadruped_standup import QuadrupedStandup

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
    task = QuadrupedStandup()

    # Set up the controller - OPTIMIZED FOR REALTIME
    ctrl = MPPI(
        task,
        num_samples=2048,       # ← Reduced from 128 (4x speedup)
        noise_level=0.3,     # ← Slightly reduced
        temperature=0.1,      # ← Higher for faster convergence
        num_randomizations=1, 
        plan_horizon=0.4,     
        spline_type="zero",
        num_knots=4,         
    )
    # ctrl = DIAL(
    #     task,
    #     num_samples=16,
    #     noise_level=0.4,
    #     beta_opt_iter=1.0,
    #     beta_horizon=1.0,
    #     temperature=0.001,
    #     plan_horizon=0.25,
    #     spline_type="zero",
    #     num_knots=11,
    #     iterations=5,
    # )
    
    
    # Define the model used for simulation - OPTIMIZED FOR REALTIME
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.01      # ← Increased from 0.01 (2x speedup)
    mj_model.opt.iterations = 1       # ← Keep minimal
    # mj_model.opt.ls_iterations = 3    # ← Reduced for speed
    mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

    # Set the initial state so the robot falls and needs to stand back up
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = mj_model.keyframe("stand").qpos
    # mj_data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0] 
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] 
    
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
            frequency=50,        # ← Reduced from 50Hz (2x speedup)
            show_traces=False,
        )
