# %%
import pickle
from datetime import datetime
import numpy as np
from include.optimisation_problem import PointBasedDesignProblem
import click
import logging
from datetime import datetime

# optimisers available
import nevergrad as ng
from cmaes import CMA
import cma as pycma

# %%
# seeds: 42, 8751, 1000047


@click.command()
@click.option(
    "--generations",
    default=2,
    type=int,
    help="number of generations",
)
@click.option(
    "--workers",
    default=2,
    type=int,
    help="number of workers for function evaluations",
)
@click.option(
    "--population",
    default=5,
    type=int,
    help="population size to be evaluated",
)
@click.option(
    "--verbosity",
    default=10,
    type=int,
    help="verbosity level, sets the terminal print frequency",
)
@click.option(
    "--algo",
    default="nevergrad_cmaes",
    type=str,
    help="choose the optimisation algorithm. Currently supported: 'cmaes' or 'fcmaes' or 'pygmo'",
)
@click.option(
    "--scenario",
    default="both",
    type=str,
    help="scenario of the simulation. Currently supported: 'rubble', 'wall', 'both' or 'None'",
)
@click.option(
    "--comment",
    default="default comment",
    type=str,
    help="comment regarding the optimisation run. will be saved along with the other details",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="set seed for all random elements in the optimisation. By default set to 0",
)
@click.option(
    "--add_base_noise",
    default=0.01,
    type=float,
    help="add noise to the CoM (x, y, z) feedback data (max 0.5 cm)",
)
@click.option(
    "--add_root_disturbance",
    default=50,
    type=float,
    help="add a disturbance to the root link, sampled uniformly between -value and value",
)
@click.option(
    "--opt_mode",
    default="co_design",
    type=str,
    help="select the optimisation mode. Currently supported: 'co_design', 'gain_tuning'",
)
@click.option(
    "--randomise_obstacle",
    default=False,
    type=bool,
    help="Randomise the location of the obstacles",
)
def main(
    generations,
    workers,
    population,
    verbosity,
    algo,
    scenario,
    comment,
    seed,
    add_base_noise,
    add_root_disturbance,
    opt_mode,
    randomise_obstacle,
):
    logging.getLogger().setLevel(logging.CRITICAL)

    dict_log = {
        "start_timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "generations": generations,
        "workers": workers,
        "population": population,
        "algo": algo,
        "scenario": scenario,
        "comment": comment,
        "seed": seed,
        "add_base_noise": add_base_noise,
        "add_root_disturbance": add_root_disturbance,
        "opt_mode": opt_mode,
        "randomise_obstacle": randomise_obstacle,
    }
    # seed numpy just in case
    np.random.seed(seed)
    meshcat = None
    opt_start_time = datetime.now().strftime("%d%m%Y%H%M%S")
    filename_progress = "../optimisation_logs/{}_progress.txt".format(opt_start_time)
    filename_opt_progress = "../optimisation_logs/{}_opt_progress.pkl".format(
        opt_start_time
    )
    filename_out = "../optimisation_logs/{}_out.pkl".format(opt_start_time)

    prob = PointBasedDesignProblem(
        seed,
        scenario,
        add_base_noise,
        add_root_disturbance,
        opt_mode,
        randomise_obstacle,
        meshcat,
    )
    prob_bounds = np.array(prob.get_bounds())
    x_mean = np.mean(prob_bounds, axis=0)

    match algo:
        case "nevergrad_cmaes":
            optimisation_progress = {}
            x0 = ng.p.Array(init=x_mean, lower=prob_bounds[0], upper=prob_bounds[1])
            opt_instance = ng.optimizers.ParametrizedCMA(
                popsize=population,
                random_init=True,
                scale=10.0,
                elitist=False,
                diagonal=True,
            )
            optimizer = opt_instance(
                parametrization=x0, budget=generations * population, num_workers=workers
            )
            optimizer.parametrization.random_state = np.random.RandomState(seed)
            for generation in range(generations):
                x_ng = [optimizer.ask() for _ in range(population)]
                x = np.array([x_ng[ii].value for ii in range(population)])
                solutions = prob.evaluate_batch_expected_fitness(x, workers)
                [optimizer.tell(x_ng[ii], solutions[ii][1]) for ii in range(population)]
                if generation % verbosity == 0:
                    ret = {"x": [], "res": []}
                    for xx, value in solutions:
                        ret["x"].append(xx)
                        ret["res"].append(value)
                    progress_file = open(filename_progress, "a")
                    progress_file.write(f"Generation: #{generation} \n")
                    for i in range(len(ret["x"])):
                        progress_file.write(
                            f"res: {ret['res'][i]:.3f} \n"
                            f"x: {np.array2string(np.array(ret['x'][i]), precision=3, separator=',')} \n"
                        )
                    progress_file.close()
                    optimisation_progress[generation] = {"x": [], "res": []}
                    optimisation_progress[generation]["x"] = ret["x"]
                    optimisation_progress[generation]["res"] = ret["res"]
                    with open(filename_opt_progress, "wb") as f:
                        pickle.dump(optimisation_progress, f)
                    with open(filename_out, "wb") as f:
                        pickle.dump(ret, f)

        case "cmaes":
            sigma = 10.0
            guess = prob.parent_rng.uniform(prob_bounds[0], prob_bounds[1])
            optimizer = CMA(
                mean=np.clip(guess, prob_bounds[0], prob_bounds[1]),
                population_size=population,
                bounds=prob_bounds.T,
                seed=seed,
                sigma=sigma,
            )
            for generation in range(generations):
                population = [optimizer.ask() for _ in range(optimizer.population_size)]
                solutions = prob.evaluate_batch_expected_fitness(population, workers)
                optimizer.tell(solutions)
                if optimizer.should_stop():
                    print("Optimisation stopping: ", optimizer.should_stop())
                    break
                if generation % verbosity == 0:
                    ret = {"x": [], "res": []}
                    for x, value in solutions:
                        ret["x"].append(x)
                        ret["res"].append(value)
                    progress_file = open(filename_progress, "a")
                    progress_file.write(f"Generation: #{generation} \n")
                    for i in range(len(ret["x"])):
                        progress_file.write(
                            f"res: {ret['res'][i]:.3f} \n"
                            f"x: {np.array2string(np.array(ret['x'][i]), precision=3, separator=',')} \n"
                        )
                    progress_file.close()
                    filename_out = "../optimisation_logs/{}_out.pkl".format(
                        opt_start_time
                    )
                    with open(filename_out, "wb") as f:
                        pickle.dump(ret, f)

        case _:
            logging.critical("Unsupported algorithm chosen")
            pass

    print("\n")
    best_idx = np.argmin(ret["res"])
    print(ret["res"][best_idx])
    print("\n")
    print(repr(ret["x"][best_idx]))
    print("\n")

    dict_log["res"] = ret["res"]
    dict_log["x"] = ret["x"]
    filename_out = "../optimisation_logs/{}_out.pkl".format(opt_start_time)

    dict_log["filename"] = filename_out
    with open(filename_out, "wb") as f:
        dict_log["end_timestamp"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        pickle.dump(dict_log, f)
    print("Saved to:", filename_out)


if __name__ == "__main__":
    main()
