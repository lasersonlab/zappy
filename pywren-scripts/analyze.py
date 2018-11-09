import glob
import matplotlib.pyplot as plt
import os
import pickle

from timeline import plot_jobs
from util import job_df_from_futures, normalize_times

logs_dir = os.path.expanduser("~/.zappy/logs")
log_files = glob.glob(os.path.join(logs_dir, "*"))
log_files.sort()

most_recent_file = log_files[-1]

print("Opening %s" % most_recent_file)

with open(most_recent_file, "rb") as file:
    indict = pickle.load(file)

    func_size_bytes = indict["futures"][0]._invoke_metadata["func_module_str_len"]
    print("Function size in bytes: %d" % func_size_bytes)

    df = job_df_from_futures(
        indict["futures"], indict["invoke_statuses"], indict["run_statuses"]
    )
    df = normalize_times(df)
    plot_jobs(df)
    plt.show()
