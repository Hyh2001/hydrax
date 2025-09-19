import os
import time
import csv
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import numpy as np
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from hydrax.task_base import Task
from hydrax.alg_base import SamplingBasedController

class Logger:
    """Logger for tracking simulation data, costs, and state variables."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = None,
        log_frequency: int = 1,  # Log every N steps
        save_frequency: int = 100,  # Save to disk every N logged steps
    ):
        """Initialize the simulation logger.
        
        Args:
            log_dir: Relative path of directory w.r.t ROOT to save logs
            experiment_name: Name of the experiment (auto-generated if None)
            log_frequency: Log data every N control steps
            save_frequency: Save data to disk every N logged data points
        """
        self.log_dir = log_dir
        self.log_frequency = log_frequency
        self.save_frequency = save_frequency
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"simulation_{timestamp}"
        self.experiment_name = experiment_name
        
        # Create directories
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Data storage
        self.data_buffer = []
        self.step_count = 0
        self.log_count = 0
        
        # Metadata
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "log_frequency": log_frequency,
            "save_frequency": save_frequency,
        }
        
        print(f"Logger initialized: {self.experiment_dir}")
    
    def log_step(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        controller: SamplingBasedController,
        task: Task,
        control: jnp.array = None,
        policy_params = None,
        plan_time: float = None,
        custom_data: Dict[str, Any] = None,
    ):
        """Log data for a single simulation step.
        
        Args:
            mj_data: MuJoCo data object
            mjx_data: MJX data object for JAX computations
            controller: The controller instance
            task: The task instance
            control: Current control inputs
            policy_params: Current policy parameters
            plan_time: Time taken for planning this step
            custom_data: Additional custom data to log
        """
        self.step_count += 1
        
        # Only log according to frequency
        if self.step_count % self.log_frequency != 0:
            return

        mjx_data = mjx.put_data(mj_model, mj_data)
        
        # Collect data
        log_entry = self._collect_data(
            mj_data, mjx_data, controller, task, control, 
            policy_params, plan_time, custom_data
        )
        
        self.data_buffer.append(log_entry)
        self.log_count += 1
        
        # Save to disk periodically
        if self.log_count % self.save_frequency == 0:
            self.save_buffer()
    
    def _collect_data(
        self,
        mj_data: mujoco.MjData,
        mjx_data: mjx.Data,
        controller: SamplingBasedController,
        task: Task,
        control: jnp.array = None,
        policy_params = None,
        plan_time: float = None,
        custom_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Collect all relevant data for logging."""
        
        log_entry = {
            "timestamp": time.time(),
            "sim_time": float(mj_data.time),
            "step": self.step_count,
        }
        
        # Task-specific costs and metrics
        if task is not None:
            log_entry.update(self._log_task_metrics(mjx_data, task, control))
        
        # Control information
        # if control is not None:
        #     log_entry.update(self._log_control(control))
        
        # Planning information
        if plan_time is not None:
            log_entry["plan_time"] = float(plan_time)
        
        # Policy parameters
        # if policy_params is not None:
        #     log_entry.update(self._log_policy_params(policy_params))
        
        # Custom data
        if custom_data is not None:
            log_entry.update(custom_data)
        
        return log_entry
    
    def _log_task_metrics(self, mjx_data: mjx.Data, task, control: jnp.array) -> Dict[str, Any]:
        """Log task-specific metrics and cost components."""
        metrics = {}
        
        try:
            # Get list of function names to call
            # Get dictionary of function names to lambda functions
            log_functions_dict = task.log_costs()
            
            # log total cost
            result = task.running_cost(mjx_data, control) + task.terminal_cost(mjx_data)
            metrics["total_cost"] = np.array(result)
            
            for metric_name, func in log_functions_dict.items():
                try:
                    if callable(func):
                        result = func(mjx_data, control)
                        metrics[metric_name] = np.array(result)
                    else:
                        raise Exception(f"Warning: {metric_name} is not a callable function or the returned results is not a saclar")
                except Exception as e:
                    raise Exception(f"Warning: Error executing metric '{metric_name}': {e}") 
        except Exception as e:
            raise Exception(f"Warning: Error logging task metrics: {e}")        
        return metrics
    
    def save_buffer(self):
        """Save the current buffer to disk."""
        if not self.data_buffer:
            return
        
        # Save as CSV
        csv_path = os.path.join(self.experiment_dir, "simulation_log.csv")
        
        # Get all unique keys from all log entries
        all_keys = set()
        for entry in self.data_buffer:
            all_keys.update(entry.keys())
        all_keys = sorted(list(all_keys))
        
        # Write CSV
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            
            if not file_exists:
                writer.writeheader()
            
            for entry in self.data_buffer:
                # Flatten list values for CSV
                flattened_entry = {}
                for key, value in entry.items():
                    if isinstance(value, list):
                        if key == "joint_positions":
                            for i, val in enumerate(value):
                                flattened_entry[f"{key}_{i}"] = val
                        elif key == "control_values":
                            for i, val in enumerate(value):
                                flattened_entry[f"control_{i}"] = val
                        else:
                            flattened_entry[key] = str(value)
                    else:
                        flattened_entry[key] = value
                
                writer.writerow(flattened_entry)
        
        # Save metadata
        metadata_path = os.path.join(self.experiment_dir, "metadata.json")
        self.metadata["last_update"] = datetime.now().isoformat()
        self.metadata["total_logged_steps"] = self.log_count
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Saved {len(self.data_buffer)} log entries to {csv_path}")
        self.data_buffer.clear()
    
    def finalize(self):
        """Save any remaining data and close the logger."""
        self.save_buffer()
        self.metadata["end_time"] = datetime.now().isoformat()
        
        # Save final metadata
        metadata_path = os.path.join(self.experiment_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Logger finalized. Total steps logged: {self.log_count}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of logged data."""
        return {
            "experiment_name": self.experiment_name,
            "total_steps": self.step_count,
            "logged_steps": self.log_count,
            "log_frequency": self.log_frequency,
            "experiment_dir": self.experiment_dir,
        }
      
    # def _log_control(self, control: np.ndarray) -> Dict[str, Any]:
    #     """Log control-related metrics."""
    #     return {
    #         "control_values": control.tolist(),
    #         "control_norm": float(np.linalg.norm(control)),
    #         "control_max": float(np.max(np.abs(control))),
    #         "control_mean": float(np.mean(control)),
    #     }
    
    # def _log_policy_params(self, policy_params) -> Dict[str, Any]:
    #     """Log policy parameter information."""
    #     metrics = {}
        
    #     try:
    #         if hasattr(policy_params, 'mean'):
    #             mean_norm = float(jnp.linalg.norm(policy_params.mean))
    #             metrics["policy_mean_norm"] = mean_norm
                
    #         if hasattr(policy_params, 'log_scale'):
    #             scale_mean = float(jnp.mean(jnp.exp(policy_params.log_scale)))
    #             metrics["policy_scale_mean"] = scale_mean
                
    #     except Exception as e:
    #         print(f"Warning: Error logging policy params: {e}")
            
    #    return metrics
    
  
class LogReader:
    """Reader for analyzing logged simulation data."""
    
    def __init__(self, experiment_path: str):
        """Initialize the log reader.
        
        Args:
            experiment_path: Path to the experiment directory or CSV file
        """
        if experiment_path.endswith('.csv'):
            # Direct path to CSV file
            self.csv_path = experiment_path
            self.experiment_dir = os.path.dirname(experiment_path)
        else:
            # Path to experiment directory
            self.experiment_dir = experiment_path
            self.csv_path = os.path.join(experiment_path, "simulation_log.csv")
        
        self.metadata_path = os.path.join(self.experiment_dir, "metadata.json")
        
        # Validate files exist
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Load data
        self._data = None
        self._metadata = None
        self._load_data()
    
    def _load_data(self):
        """Load the CSV data and metadata."""
        try:
            # Load CSV data
            import pandas as pd
            self._data = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self._data)} log entries from {self.csv_path}")
            
        except ImportError:
            # Fallback to basic CSV reading if pandas not available
            self._data = []
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric strings back to numbers
                    converted_row = {}
                    for key, value in row.items():
                        try:
                            # Try to convert to float
                            converted_row[key] = float(value)
                        except (ValueError, TypeError):
                            # Keep as string if conversion fails
                            converted_row[key] = value
                    self._data.append(converted_row)
            
            print(f"Loaded {len(self._data)} log entries from {self.csv_path}")
        
        # Load metadata if available
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}
    
    @property
    def data(self):
        """Get the raw data."""
        return self._data
    
    @property
    def metadata(self):
        """Get the experiment metadata."""
        return self._metadata
    
    def get_column_names(self) -> List[str]:
        """Get all available column names."""
        if hasattr(self._data, 'columns'):  # pandas DataFrame
            return list(self._data.columns)
        else:  # list of dicts
            if self._data:
                return list(self._data[0].keys())
            return []
    
    def get_time_series(self, column: str) -> tuple:
        """Get time series data for a specific column.
        
        Args:
            column: Column name to extract
            
        Returns:
            tuple: (time_values, column_values)
        """
        if hasattr(self._data, 'columns'):  # pandas DataFrame
            if 'sim_time' in self._data.columns and column in self._data.columns:
                return self._data['sim_time'].values, self._data[column].values
            else:
                raise KeyError(f"Column '{column}' or 'sim_time' not found")
        else:  # list of dicts
            times = [row.get('sim_time', 0) for row in self._data]
            values = [row.get(column, 0) for row in self._data]
            return np.array(times), np.array(values)
    
    def get_multiple_series(self, columns: List[str]) -> Dict[str, tuple]:
        """Get multiple time series at once.
        
        Args:
            columns: List of column names
            
        Returns:
            Dict mapping column names to (time, values) tuples
        """
        result = {}
        for col in columns:
            try:
                result[col] = self.get_time_series(col)
            except KeyError:
                print(f"Warning: Column '{col}' not found")
        return result
    
    def get_cost_column_names(self) -> List[str]:
        """Get all cost-related column names."""
        return [col for col in self.get_column_names() 
                if 'cost' in col.lower()]
    
    def get_cost_breakdown(self) -> Dict[str, tuple]:
        """Get all cost-related time series."""
        cost_columns = self.get_cost_column_names()
        return self.get_multiple_series(cost_columns)
    
    def plot_time_series(self, columns: List[str], save_path: str = None):
        """Plot time series for given columns.
        
        Args:
            columns: List of column names to plot
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(len(columns), 1, figsize=(12, 3*len(columns)), 
                                   sharex=True)
            if len(columns) == 1:
                axes = [axes]
            
            for i, col in enumerate(columns):
                try:
                    times, values = self.get_time_series(col)
                    axes[i].plot(times, values, label=col)
                    axes[i].set_ylabel(col)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].legend()
                except KeyError:
                    axes[i].text(0.5, 0.5, f"Column '{col}' not found", 
                               ha='center', va='center', transform=axes[i].transAxes)
            
            axes[-1].set_xlabel('Simulation Time (s)')
            plt.suptitle(f"Time Series - {self._metadata.get('experiment_name', 'Unknown')}")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available. Cannot create plots.")
    
    def print_info(self):
        """Print information about the logged data."""
        print(f"\n=== Log Reader Info ===")
        print(f"Experiment: {self._metadata.get('experiment_name', 'Unknown')}")
        print(f"Data points: {len(self._data)}")
        print(f"Columns: {len(self.get_column_names())}")
        
        if 'start_time' in self._metadata:
            print(f"Start time: {self._metadata['start_time']}")
        if 'end_time' in self._metadata:
            print(f"End time: {self._metadata['end_time']}")
        
        print(f"\nAvailable columns:")
        columns = self.get_column_names()
        for i in range(0, len(columns), 4):  # Print 4 columns per line
            line_cols = columns[i:i+4]
            print(f"  {', '.join(line_cols)}")
        
        # Show simulation time range
        if 'sim_time' in columns:
            times, _ = self.get_time_series('sim_time')
            print(f"\nSimulation time range: {times[0]:.3f} - {times[-1]:.3f} seconds")
            print(f"Total duration: {times[-1] - times[0]:.3f} seconds")
    
    @classmethod
    def find_experiments(cls, log_dir: str) -> List[str]:
        """Find all experiment directories in a log directory.
        
        Args:
            log_dir: Directory to search for experiments
            
        Returns:
            List of experiment directory paths
        """
        experiments = []
        if os.path.exists(log_dir):
            for item in os.listdir(log_dir):
                exp_path = os.path.join(log_dir, item)
                if os.path.isdir(exp_path):
                    csv_path = os.path.join(exp_path, "simulation_log.csv")
                    if os.path.exists(csv_path):
                        experiments.append(exp_path)
        return sorted(experiments)
    
    def filter_by_time(self, start_time: float = None, end_time: float = None):
        """Filter data by simulation time range.
        
        Args:
            start_time: Start time (inclusive)
            end_time: End time (inclusive)
            
        Returns:
            New LogReader instance with filtered data
        """
        if hasattr(self._data, 'query'):  # pandas DataFrame
            query_parts = []
            if start_time is not None:
                query_parts.append(f"sim_time >= {start_time}")
            if end_time is not None:
                query_parts.append(f"sim_time <= {end_time}")
            
            if query_parts:
                filtered_data = self._data.query(" and ".join(query_parts))
            else:
                filtered_data = self._data.copy()
        else:  # list of dicts
            filtered_data = []
            for row in self._data:
                sim_time = row.get('sim_time', 0)
                if start_time is not None and sim_time < start_time:
                    continue
                if end_time is not None and sim_time > end_time:
                    continue
                filtered_data.append(row.copy())
        
        # Create new LogReader instance with filtered data
        filtered_reader = LogReader.__new__(LogReader)
        filtered_reader.experiment_dir = self.experiment_dir
        filtered_reader.csv_path = self.csv_path
        filtered_reader.metadata_path = self.metadata_path
        filtered_reader._data = filtered_data
        filtered_reader._metadata = self._metadata.copy()
        
        return filtered_reader
 
    # def get_state_variables(self) -> Dict[str, tuple]:
    #     """Get state variable time series."""
    #     state_columns = [col for col in self.get_column_names() 
    #                     if any(keyword in col.lower() for keyword in 
    #                           ['pos', 'vel', 'height', 'orientation', 'angle'])]
    #     return self.get_multiple_series(state_columns)
    
    # def get_statistics(self, column: str) -> Dict[str, float]:
    #     """Get basic statistics for a column.
        
    #     Args:
    #         column: Column name
            
    #     Returns:
    #         Dictionary with mean, std, min, max, etc.
    #     """
    #     _, values = self.get_time_series(column)
        
    #     return {
    #         'mean': float(np.mean(values)),
    #         'std': float(np.std(values)),
    #         'min': float(np.min(values)),
    #         'max': float(np.max(values)),
    #         'median': float(np.median(values)),
    #         'count': len(values),
    #     }
    
    # def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
    #     """Get statistics for all numeric columns."""
    #     numeric_columns = []
        
    #     if hasattr(self._data, 'select_dtypes'):  # pandas DataFrame
    #         numeric_columns = list(self._data.select_dtypes(include=[np.number]).columns)
    #     else:  # list of dicts
    #         # Find numeric columns by checking first few rows
    #         sample_size = min(10, len(self._data))
    #         for col in self.get_column_names():
    #             try:
    #                 values = [self._data[i][col] for i in range(sample_size)]
    #                 # Check if all values are numeric
    #                 if all(isinstance(v, (int, float)) for v in values):
    #                     numeric_columns.append(col)
    #             except (KeyError, IndexError):
    #                 continue
        
    #     return {col: self.get_statistics(col) for col in numeric_columns}
   
    # def export_subset(self, columns: List[str], output_path: str):
    #     """Export a subset of columns to a new CSV file.
        
    #     Args:
    #         columns: List of column names to export
    #         output_path: Path for the output CSV file
    #     """
    #     if hasattr(self._data, 'to_csv'):  # pandas DataFrame
    #         self._data[columns].to_csv(output_path, index=False)
    #     else:  # list of dicts
    #         with open(output_path, 'w', newline='') as f:
    #             writer = csv.DictWriter(f, fieldnames=columns)
    #             writer.writeheader()
    #             for row in self._data:
    #                 filtered_row = {col: row.get(col, '') for col in columns}
    #                 writer.writerow(filtered_row)
        
    #     print(f"Exported {len(columns)} columns to {output_path}")
    
