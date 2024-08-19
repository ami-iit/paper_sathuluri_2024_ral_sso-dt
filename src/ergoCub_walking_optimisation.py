# %%
import argparse
import logging
import pickle
from datetime import datetime
import numpy as np

from cmaes import CMA

from pydrake.geometry import StartMeshcat
from include.optimisation_problem import GainTuningProblem
from include.utils import find_running_as_notebook
from include.simulation_system import run_walking_sim

running_as_notebook = find_running_as_notebook()

# %%

if not running_as_notebook:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations",
        help="number of generations",
        nargs="?",
        const=2,
        default=2,
        type=int,
    )
    parser.add_argument(
        "--workers",
        help="number of workers for function evaluations",
        nargs="?",
        const=2,
        default=2,
        type=int,
    )
    parser.add_argument(
        "--population",
        help="population size to be evaluated",
        nargs="?",
        const=5,
        default=5,
        type=int,
    )
    parser.add_argument(
        "--verbosity",
        help="verbosity level, sets the terminal print frequency",
        nargs="?",
        const=10,
        default=10,
        type=int,
    )
    parser.add_argument(
        "--algo",
        help="choose the optimisation algorithm. Currently supported: 'cmaes' or 'fcmaes' or 'pygmo' ",
        nargs="?",
        const="cmaes",
        default="cmaes",
        type=str,
    )
    parser.add_argument(
        "--scenario",
        help="scenario of the simulation. Currently supported: 'rubble', 'wall', 'both' or 'None' ",
        nargs="?",
        const="None",
        default="None",
        type=str,
    )
    parser.add_argument(
        "--comment",
        help="comment regarding the optimisation run. will be saved along with the other details",
        nargs="?",
        const="None",
        default="default comment",
        type=str,
    )
    parser.add_argument(
        "--seed",
        help="set seed for all random elements in the optimisation. By default set to 0",
        nargs="?",
        const=0,
        default=0,
        type=int,
    )
    parser.add_argument(
        "--add_base_noise",
        help="add noise to the CoM (x, y, z) feedback data (max 0.5 cm) ",
        nargs="?",
        const=0,
        default=0,
        type=float,
    )
    parser.add_argument(
        "--add_root_disturbance",
        help="add a disturbance to the root link, sampled uniformly between -value and value",
        nargs="?",
        const=0,
        default=0,
        type=float,
    )
    parser.add_argument(
        "--opt_mode",
        help="select the optimisation mode. Currently supported: 'co_design', 'gain_tuning'",
        nargs="?",
        const="gain_tuning",
        default="gain_tuning",
        type=str,
    )
    parser.add_argument(
        "--compute_expectation",
        help="Change the objective to an expectation computed over N trials",
        nargs="?",
        const=1,
        default=1,
        type=int,
    )
    parser.add_argument(
        "--randomise_obstacle",
        help="Randomise the location of the obstacles",
        nargs="?",
        const=False,
        default=False,
        type=bool,
    )

    parser.add_argument(
        "--robot_type",
        help="Select the motor model to be loaded. Currently supported: 'QDD', 'harmonic'",
        nargs="?",
        const="QDD",
        default="QDD",
        type=str,
    )

    args = parser.parse_args()

    generations = int(args.generations)
    workers = int(args.workers)
    population = int(args.population)
    verbosity = int(args.verbosity)
    algo = str(args.algo)
    scenario = str(args.scenario)
    comment = str(args.comment)
    seed = int(args.seed)
    add_base_noise = float(args.add_base_noise)
    add_root_disturbance = float(args.add_root_disturbance)
    opt_mode = str(args.opt_mode)
    compute_expectation = int(args.compute_expectation)
    randomise_obstacle = bool(args.randomise_obstacle)
    robot_type = str(args.robot_type)

    logging.getLogger().setLevel(logging.CRITICAL)

    dict_log = {}
    dict_log["start_timestamp"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    dict_log["generations"] = generations
    dict_log["workers"] = workers
    dict_log["population"] = population
    dict_log["algo"] = algo
    dict_log["scenario"] = scenario
    dict_log["comment"] = comment
    dict_log["seed"] = seed
    dict_log["add_base_noise"] = add_base_noise
    dict_log["add_root_disturbance"] = add_root_disturbance
    dict_log["opt_mode"] = opt_mode
    dict_log["randomise_obstacle"] = randomise_obstacle
    dict_log["robot_type"] = robot_type
    dict_log["compute_expectation"] = compute_expectation

    meshcat = None
    viz = None


# %%
if __name__ == "__main__" and not running_as_notebook:
    opt_start_time = datetime.now().strftime("%d%m%Y%H%M%S")
    filename_progress = "../optimisation_logs/{}_progress.txt".format(opt_start_time)
    progress_file = open(filename_progress, "w")
    start_time = str(datetime.now())
    progress_file.write("Execution started at: " + start_time + "\n")
    progress_file.close()
    prob = GainTuningProblem(
        seed,
        scenario,
        add_base_noise,
        add_root_disturbance,
        opt_mode,
        randomise_obstacle,
        robot_type,
        meshcat,
    )
    match algo:
        case "pygmo":
            pass

        case "fcmaes":
            pass

        case "cmaes":
            sigma = 1.0
            prob_bounds = np.asarray(prob.get_bounds())
            pkl_filename_history = "12032024204049_out"
            with open(
                "../optimisation_logs/{}.pkl".format(pkl_filename_history), "rb"
            ) as f:
                expt_data = pickle.load(f)
                guess = expt_data["x"][np.argmin(expt_data["res"])]

            optimizer = CMA(
                mean=np.clip(guess, prob_bounds[0], prob_bounds[1]),
                population_size=population,
                bounds=prob_bounds.T,
                seed=1402481612,  # secrets.randbits(32)
                sigma=sigma,
            )

            for generation in range(generations):
                population = [optimizer.ask() for _ in range(optimizer.population_size)]
                solutions = prob.evaluate_batch_expected_fitness(
                    population, workers, compute_expectation
                )
                optimizer.tell(solutions)
                if optimizer.should_stop():
                    print("Optimisation stopping: ", optimizer.should_stop())
                    break
                if generation % verbosity == 0:
                    sol = min(solutions, key=lambda t: t[1])
                    progress_file = open(filename_progress, "a")
                    progress_file.write(
                        f"Generation: #{generation} \n {sol[1]} \n {repr(sol[0])} \n"
                    )
                    progress_file.close()

                    ret = {"x": [], "res": []}
                    for x, value in solutions:
                        ret["x"].append(x)
                        ret["res"].append(value)
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
        pickle.dump(dict_log, f)
    print("Saved to:", filename_out)

    end_time = str(datetime.now())
    progress_file = open(filename_progress, "a")
    progress_file.write("Execution ended at: " + end_time + "\n")
    progress_file.close()

# %%
if running_as_notebook:
    meshcat = StartMeshcat()
    save_meshcat = meshcat
    rng = np.random.RandomState(0)


# %%
if running_as_notebook:
    print("---------------------------------")
    print("Experiment details")
    print("---------------------------------")

    run_type = "saved_file"

    match run_type:
        case "saved_file":
            file_tag = "12032024204049_out"
            filename = "../optimisation_logs/{}.pkl".format(file_tag)
            with open(filename, "rb") as f:
                expt_data = pickle.load(f)

            seed = expt_data["seed"]
            add_base_noise = expt_data["add_base_noise"]
            add_root_disturbance = expt_data["add_root_disturbance"]
            opt_mode = expt_data["opt_mode"]
            randomise_obstacle = expt_data["randomise_obstacle"]
            robot_type = expt_data["robot_type"]
            scenario = expt_data["scenario"]
            workers = expt_data["workers"]
            compute_expectation = expt_data["compute_expectation"]

            print("Start Timestamp:", expt_data["start_timestamp"])
            print("Comment:", expt_data["comment"])
            print("Generations:", expt_data["generations"])
            print("Workers:", expt_data["workers"])
            print("Population:", expt_data["population"])
            print("Best fval:", np.min(expt_data["res"]))
            test_x = expt_data["x"][np.argmin(expt_data["res"])]
            print("Best gains:", test_x)
            print("Seed:", expt_data["seed"])
            print("algo:", expt_data["algo"])

    prob = GainTuningProblem(
        seed,
        scenario,
        add_base_noise,
        add_root_disturbance,
        opt_mode,
        randomise_obstacle,
        robot_type,
        meshcat,
    )

    test_x = np.clip(test_x, prob.get_bounds()[0], prob.get_bounds()[1])

    print("gains and weights:", test_x[:28])
    print("gear ratios:", test_x[28 : 28 + 13])
    print("motor masses:", test_x[28 + 13 :])

    Tf, tau_mlog, tau_flog, mv_log, motor_constant_matrix = run_walking_sim(
        test_x,
        scenario,
        rng,
        add_base_noise,
        add_root_disturbance,
        opt_mode,
        randomise_obstacle,
        robot_type,
        meshcat,
        time_step=1e-3,
    )

    Tf, P_f, P_j = prob.compute_qois(
        Tf, tau_mlog, tau_flog, mv_log, motor_constant_matrix
    )
    print("Sim time:", Tf)
    print("Friction loss:", P_f)
    print("Joule loss:", P_j)
    metric = prob.compute_metric(Tf, P_f, P_j, test_x)
    print("Metric:", metric)

# %%
