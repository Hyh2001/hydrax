from hydrax.utils.logger import LogReader
from hydrax import ROOT

reader = LogReader(ROOT + "/logs/simulation_20250922_171152")

# Get basic info
reader.print_info()

column_names = reader.get_column_names()
# column_names = reader.get_cost_column_names()

reader.plot_time_series(column_names)