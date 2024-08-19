# %%

import numpy as np
import pickle
import nevergrad as ng
from datetime import datetime
from include.optimisation_problem import SetBasedCoDesignProblem
import click
from tqdm import tqdm


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
    "--nas",
    default=100,
    type=int,
    help="num a space samples",
)
@click.option(
    "--nbs",
    default=50,
    type=int,
    help="num b space samples",
)
@click.option(
    "--brange",
    default=0.5,
    type=float,
    help="range around point optimum to sample for b space",
)
@click.option(
    "--amargin",
    default=1,
    type=float,
    help="range around point optimum to sample for b space",
)
@click.option(
    "--b_space_workers",
    default=10,
    type=float,
    help="workers for bspace sims",
)
@click.option(
    "--use_surrogate",
    default=False,
    type=bool,
)
@click.option(
    "--use_compensation",
    default=False,
    type=bool,
)
@click.option(
    "--min_a_space_threshold",
    default=0,
    type=float,
)
@click.option(
    "--opt_type",
    default="ssdt",
    type=str,
)
def main(
    generations,
    workers,
    population,
    verbosity,
    algo,
    comment,
    seed,
    nas,
    nbs,
    brange,
    amargin,
    b_space_workers,
    use_surrogate,
    use_compensation,
    min_a_space_threshold,
    opt_type,
):

    dict_log = {
        "start_timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "generations": generations,
        "workers": workers,
        "population": population,
        "comment": comment,
        "seed": seed,
        "nas": nas,
        "nbs": nbs,
        "brange": brange,
        "algo": algo,
        "amargin": amargin,
        "b_space_workers": b_space_workers,
        "use_surrogate": use_surrogate,
        "use_compensation": use_compensation,
        "min_a_space_threshold": min_a_space_threshold,
        "opt_type": opt_type,
    }

    np.random.seed(seed)
    opt_start_time = datetime.now().strftime("%d%m%Y%H%M%S")
    filename_progress = "../optimisation_logs/omega/SSDT/{}_progress.txt".format(
        opt_start_time
    )
    filename_opt_progress = (
        "../optimisation_logs/omega/SSDT/{}_opt_progress.pkl".format(opt_start_time)
    )
    filename_out = "../optimisation_logs/omega/SSDT/{}_out.pkl".format(opt_start_time)

    selected_point_optimum, classifier_path = setup_problem()

    SSEprob = SetBasedCoDesignProblem(
        seed=seed,
        search_margin=amargin,
        num_a_space_samples=nas,
        num_b_space_samples=nbs,
        b_space_sample_range=brange,
        use_surrogate=use_surrogate,
        use_compensation=use_compensation,
        b_space_workers=b_space_workers,
        min_a_space_threshold=min_a_space_threshold,
        opt_type=opt_type,
    )
    print("Using surrogate:", SSEprob.use_surrogate)
    print("Using compensation:", SSEprob.use_compensation)

    SSEprob.set_point_optimum_init(selected_point_optimum)
    if use_surrogate:
        SSEprob.load_classifier(classifier_path)

    prob_bounds = np.array(SSEprob.get_bounds())
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
            for generation in tqdm(range(generations), desc="Generations"):
                x_ng = [optimizer.ask() for _ in range(population)]
                x = np.array([x_ng[ii].value for ii in range(population)])
                if SSEprob.use_surrogate:
                    solutions = SSEprob.evaluate_batch_expected_fitness(x, workers)
                else:
                    solutions = []
                    for xi in x:
                        soli = SSEprob.fitness(xi)
                        solutions.append((xi, soli))
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
                            f"res: {ret['res'][i]:.5f} \n"
                            f"x: {np.array2string(np.array(ret['x'][i]), precision=5, separator=',')} \n"
                        )
                    progress_file.close()

                    # for opt progress
                    optimisation_progress[generation] = {"x": [], "res": []}
                    optimisation_progress[generation]["x"] = ret["x"]
                    optimisation_progress[generation]["res"] = ret["res"]
                    with open(filename_opt_progress, "wb") as f:
                        pickle.dump(optimisation_progress, f)

                    with open(filename_out, "wb") as f:
                        pickle.dump(ret, f)

    print("\n")
    best_idx = np.argmin(ret["res"])
    print(ret["res"][best_idx])
    print("\n")
    print(repr(ret["x"][best_idx]))
    print("\n")

    dict_log["res"] = ret["res"]
    dict_log["x"] = ret["x"]
    filename_out = "../optimisation_logs/omega/SSDT/{}_out.pkl".format(opt_start_time)

    dict_log["filename"] = filename_out
    with open(filename_out, "wb") as f:
        dict_log["end_timestamp"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        pickle.dump(dict_log, f)
    print("Saved to:", filename_out)


# %%


def setup_problem():
    filetag = "18012025032840"
    with open(f"../optimisation_logs/omega/{filetag}_opt_progress.pkl", "rb") as f:
        all_sols = pickle.load(f)
    all_res = np.array([all_sols[k]["res"] for k in all_sols.keys()])
    all_res = all_res.flatten()
    all_x = np.array([all_sols[k]["x"] for k in all_sols.keys()])
    all_x = all_x.reshape(-1, all_x.shape[-1])
    selected_point_optimum = all_x[np.argmin(all_res)]
    print("gains and weights:", selected_point_optimum[:28])
    print("gear ratios:", selected_point_optimum[28 : 28 + 13])
    print("motor masses:", selected_point_optimum[28 + 13 :])
    classifier_path = "../optimisation_logs/omega"
    return selected_point_optimum, classifier_path

    # %%


if __name__ == "__main__":
    main()
