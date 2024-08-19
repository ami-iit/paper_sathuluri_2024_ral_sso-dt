# %%
from include.simulation_system import run_walking_sim
import numpy as np
from pathlib import Path
import pickle
from pydrake.geometry import StartMeshcat
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Computer Modern Roman"
plt.rcParams["axes.grid"] = True
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["figure.dpi"] = 600

# %% Load the saved DV
log_dir = Path("../optimisation_logs").resolve().__str__()
meshcat = StartMeshcat()

# %%


def load_result_files(pkl_filename, prog_filename):
    pkl_filename = "../optimisation_logs/{}.pkl".format(pkl_filename)
    with open(pkl_filename, "rb") as f:
        expt_data = pickle.load(f)
        # nicely formatted print of the data
        print("+" + "-" * 20 + "+" + "-" * 30 + "+")
        print("|{:<20s}|{:<30s}|".format("Key", "Value"))
        print("+" + "-" * 20 + "+" + "-" * 30 + "+")
        for key, value in expt_data.items():
            if key != "res" and key != "x":
                print("|{:<20s}|{:<30s}|".format(key, str(value)))
        print("+" + "-" * 20 + "+" + "-" * 30 + "+")

    prog_filename = "../optimisation_logs/{}.txt".format(prog_filename)
    progress_text = open(prog_filename, "r")
    lines = progress_text.readlines()

    gens = []
    xs = []
    fvals = []

    for ii in range(len(lines)):
        if "Generation: #" in lines[ii]:
            gen = int(lines[ii].split(": #")[1])
            gens.append(gen)
            fvals.append(float(lines[ii + 1]))

    return expt_data, gens, xs, fvals


def plot_optimisation_progress(pkl_filename, prog_filename):
    expt_data, gens, xs, fvals = load_result_files(pkl_filename, prog_filename)

    fig, ax = plt.subplots()

    ax.plot(gens, fvals, color="black")
    ax.plot(gens, fvals, "x", color="red", linewidth=0.5, markersize=5)

    ax.set_xlabel("Generations")
    ax.set_ylabel("Objective function value")

    plt.show()
    return gens, xs, fvals


# %%

case_dict = {
    "seed": 1468416815,
    "add_base_noise": 0.01,
    "add_root_disturbance": 1000.0,
    "randomise_obstacle": 0,
    "scenario": "both",
    "opt_mode": "co_design",
    "robot_type": "harmonic",
    "meshcat": meshcat,
}

pkl_filename_history = "12032024204049_out"
prog_filename_history = "12032024170357_progress"

gens, xs, fvals = plot_optimisation_progress(
    pkl_filename_history, prog_filename_history
)

expt_data_history, _, _, _ = load_result_files(
    pkl_filename_history, prog_filename_history
)

best_x = expt_data_history["x"][np.argmin(expt_data_history["res"])]


# %%
Tf, tau_mlog, tau_flog, mv_log, motor_constant_matrix = run_walking_sim(
    best_x,
    rng=np.random.default_rng(case_dict["seed"]),
    scenario=case_dict["scenario"],
    add_base_noise=case_dict["add_base_noise"],
    add_root_disturbance=case_dict["add_root_disturbance"],
    opt_mode=case_dict["opt_mode"],
    randomise_obstacle=case_dict["randomise_obstacle"],
    robot_type=case_dict["robot_type"],
    meshcat=case_dict["meshcat"],
    time_step=1e-4,
)

# %%
