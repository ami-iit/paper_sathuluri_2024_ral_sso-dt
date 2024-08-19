# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from include.optimisation_problem import PointBasedDesignProblem
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
import os
from include.utils import load_solution, plot_optimisation_progress


meshcat = None
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


# %%
sim_config = {
    "seed": 42,
    "add_base_noise": 0.05,
    "add_root_disturbance": 300,
    "opt_mode": "co_design",
    "meshcat": meshcat,
    "time_step": 0.5e-3,
}

filetag = "omega/18012025032840"
selected_point_optimum = load_solution(filetag)
print("gains:", selected_point_optimum[:28])
print("gear ratios:", selected_point_optimum[28 : 28 + 13])
print("motor masses:", selected_point_optimum[28 + 13 :])
gens, xs, fvals = plot_optimisation_progress(filetag)


# %%

prob = PointBasedDesignProblem(
    seed=sim_config["seed"],
    add_base_noise=sim_config["add_base_noise"],
    add_root_disturbance=sim_config["add_root_disturbance"],
    opt_mode=sim_config["opt_mode"],
    meshcat=sim_config["meshcat"],
    time_step=sim_config["time_step"],
)


# %%
def sample_fixed_offset_point_optimum(
    batch_size=128, sample_size=int(1e3 * 128), m=0.12
):
    zeta_samples = prob.parent_rng.uniform(
        [-sim_config["add_root_disturbance"], -sim_config["add_base_noise"]],
        [sim_config["add_root_disturbance"], sim_config["add_base_noise"]],
        (sample_size, 2),
    )
    fixed_offset = np.array([25] * 28 + [40] * 13 + [0.5] * 13)
    space_lb = np.clip(
        selected_point_optimum - fixed_offset,
        prob.get_bounds()[0],
        prob.get_bounds()[1],
    )
    space_ub = np.clip(
        selected_point_optimum + fixed_offset,
        prob.get_bounds()[0],
        prob.get_bounds()[1],
    )
    space_bounds = np.vstack((space_lb, space_ub))
    design_samples = prob.parent_rng.uniform(
        space_bounds[0], space_bounds[1], (sample_size, 54)
    )
    combined_samples = np.concatenate((design_samples, zeta_samples), axis=1)

    def sim_out(i):
        Tf, P_f, P_j, metric = prob.compute_fitness_and_components_with_noise(
            combined_samples[i][:54],
            combined_samples[i][-1],
            combined_samples[i][-2],
            prob.parent_rng,
        )

        return np.concatenate(
            (
                combined_samples[i],
                [Tf],
                [P_f],
                [P_j],
                [metric],
            )
        )

    out_folder = f"../optimisation_logs/{filetag}_collected_samples_{m}/{datetime.now().strftime('%H%M%S')}/"
    os.makedirs(out_folder, exist_ok=True)
    for start in tqdm(range(0, sample_size, batch_size)):
        end = min(start + batch_size, sample_size)
        chunk = Parallel(n_jobs=-1)(delayed(sim_out)(i) for i in range(start, end))
        with open(
            out_folder + f"/samples_{int(end/batch_size)}.pkl",
            "ab",
        ) as f:
            pickle.dump(chunk, f)


sample_fixed_offset_point_optimum(
    batch_size=128, sample_size=int(1e3 * 128), m="offset"
)

# %%
