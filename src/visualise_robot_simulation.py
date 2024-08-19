# %%
import numpy as np
from pydrake.geometry import StartMeshcat
import matplotlib.pyplot as plt
from include.optimisation_problem import PointBasedDesignProblem
from include.utils import load_solution

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

meshcat = StartMeshcat()


filetag = "../optimisation_logs/omega/18012025032840"
selected_point_optimum = load_solution(filetag)
print("gains:", selected_point_optimum[:28])
print("gear ratios:", selected_point_optimum[28 : 28 + 13])
print("motor masses:", selected_point_optimum[28 + 13 :])
sim_config = {
    "seed": 42,
    "add_base_noise": 0.02,
    "add_root_disturbance": 100,
    "opt_mode": "co_design",
    "meshcat": meshcat,
    "time_step": 1e-3,
}

prob = PointBasedDesignProblem(
    seed=sim_config["seed"],
    add_base_noise=sim_config["add_base_noise"],
    add_root_disturbance=sim_config["add_root_disturbance"],
    opt_mode=sim_config["opt_mode"],
    meshcat=sim_config["meshcat"],
    time_step=sim_config["time_step"],
)


Tf, P_f, P_j, metric = prob.compute_fitness_and_components(
    selected_point_optimum, prob.parent_rng
)

print("Sim time:", Tf)
print("Friction loss:", P_f)
print("Joule loss:", P_j)
print("Metric:", metric)

# %%
