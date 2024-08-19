from joblib import Parallel, delayed
import numpy as np
from include.simulation_system import run_walking_sim
from tqdm import tqdm
import pickle


class PointBasedDesignProblem:
    def __init__(
        self,
        seed=0,
        scenario="both",
        add_base_noise=0,
        add_root_disturbance=0,
        opt_mode="co_design",
        randomise_obstacle=False,
        meshcat=None,
        time_step=1e-3,
    ):
        self.parent_rng = np.random.default_rng(seed)
        self.scenario = scenario
        self.add_base_noise = add_base_noise
        self.add_root_disturbance = add_root_disturbance
        self.opt_mode = opt_mode
        self.randomise_obstacle = randomise_obstacle
        self.meshcat = meshcat
        self.time_step = time_step
        pass

    def compute_metric(
        self,
        Tf,
        P_f,
        P_j,
        x=None,
        T_limit=5,
        penal_val=1e3,
    ):
        weight_P_f = 1
        weight_P_j = 1
        if Tf > T_limit:
            result = weight_P_f * P_f + weight_P_j * P_j + penal_val * (21 - Tf)
        else:
            result = weight_P_f * 5e3 + weight_P_j * 5e3 + penal_val * (21 - Tf)
        return result

    def compute_qois(
        self,
        Tf,
        tau_mlog,
        tau_flog,
        mv_log,
        motor_constant_matrix,
        max_clip=1e3,
    ):
        num_samples = len(mv_log.sample_times())
        tau_mdat = np.nan_to_num(
            np.clip(tau_mlog.data(), -max_clip, max_clip), nan=max_clip
        )
        tau_fdat = np.nan_to_num(
            np.clip(tau_flog.data(), -max_clip, max_clip),
            nan=max_clip,
        )
        mvdat = np.nan_to_num(np.clip(mv_log.data(), -max_clip, max_clip), nan=max_clip)
        P_f = np.sum(np.abs(tau_fdat * mvdat)) / num_samples
        P_j = np.sum((tau_mdat.T @ motor_constant_matrix).T * tau_mdat) / num_samples
        return Tf, P_f, P_j

    def fitness(self, x, child_rng=None):
        if child_rng is None:
            child_rng = self.parent_rng
        (
            Tf,
            tau_mlog,
            tau_flog,
            mv_log,
            motor_constant_matrix,
        ) = run_walking_sim(
            x,
            self.scenario,
            child_rng,
            self.add_base_noise,
            self.add_root_disturbance,
            self.opt_mode,
            self.randomise_obstacle,
            self.meshcat,
            self.time_step,
        )
        Tf, P_f, P_j = self.compute_qois(
            Tf, tau_mlog, tau_flog, mv_log, motor_constant_matrix
        )
        return self.compute_metric(Tf, P_f, P_j, x)

    def compute_fitness_and_components(self, x, child_rng):
        (
            Tf,
            tau_mlog,
            tau_flog,
            mv_log,
            motor_constant_matrix,
        ) = run_walking_sim(
            x,
            self.scenario,
            child_rng,
            self.add_base_noise,
            self.add_root_disturbance,
            self.opt_mode,
            self.randomise_obstacle,
            self.meshcat,
            self.time_step,
        )
        Tf, P_f, P_j = self.compute_qois(
            Tf, tau_mlog, tau_flog, mv_log, motor_constant_matrix
        )
        return Tf, P_f, P_j, self.compute_metric(Tf, P_f, P_j, x)

    def compute_fitness_and_components_with_noise(
        self, x, base_noise, root_disturbance, child_rng
    ):
        (
            Tf,
            tau_mlog,
            tau_flog,
            mv_log,
            motor_constant_matrix,
        ) = run_walking_sim(
            x,
            self.scenario,
            child_rng,
            base_noise,
            root_disturbance,
            self.opt_mode,
            self.randomise_obstacle,
            self.meshcat,
            time_step=self.time_step,
        )
        Tf, P_f, P_j = self.compute_qois(
            Tf, tau_mlog, tau_flog, mv_log, motor_constant_matrix
        )
        return Tf, P_f, P_j, self.compute_metric(Tf, P_f, P_j, x)

    def evaluate_batch_expected_fitness(self, population, workers):
        temp_solutions = []
        child_rngs = self.parent_rng.spawn(len(population))

        temp_solutions.append(
            Parallel(n_jobs=workers)(
                delayed(self.fitness)(population[ii], child_rngs[ii])
                for ii in range(len(population))
            )
        )
        temp_solutions = temp_solutions[0]
        solutions = []
        for ii in range(len(population)):
            solutions.append((population[ii], temp_solutions[ii]))
        return solutions

    def get_bounds(self):
        match self.opt_mode:
            case "co_design":
                return (
                    [0.5] * 28 + [50] * 13 + [0.1] * 13,
                    [500] * 28 + [160] * 13 + [1.5] * 13,
                )
            case "gain_tuning":
                return ([0.5] * 28, [500] * 28)
            case _:
                pass

    def batch_fitness(self, batch_x):
        batch_fitness = np.asarray(
            Parallel(n_jobs=self.workers)(
                delayed(self.fitness)(batch_x[ii]) for ii in range(len(batch_x))
            )
        ).flatten()
        return batch_fitness

    def has_batch_fitness(self):
        return True


class SetBasedCoDesignProblem(PointBasedDesignProblem):
    def __init__(
        self,
        seed=42,
        add_base_noise=0,
        add_root_disturbance=0,
        opt_mode="co_design",
        meshcat=None,
        search_margin=0.12,
        num_a_space_samples=50,
        num_b_space_samples=50,
        b_space_sample_range=0.5,
        use_surrogate=True,
        use_compensation=True,
        b_space_workers=10,
        min_a_space_threshold=1.1e-2,
        opt_type="ssdt",
    ):
        super().__init__(
            seed=seed,
            add_base_noise=add_base_noise,
            add_root_disturbance=add_root_disturbance,
            opt_mode=opt_mode,
            meshcat=meshcat,
        )
        self.search_margin = search_margin
        self.use_compensation = use_compensation
        self.compute_expectation = 0
        self.init_x = None
        self.max_fitness = 31000
        self.a_space_size_weight = 1
        self.b_space_size_weight = 1
        self.num_b_space_samples = num_b_space_samples
        self.b_space_sample_range = b_space_sample_range
        self.num_a_space_samples = num_a_space_samples
        self.use_surrogate = use_surrogate
        self.a_space_dim = 13 + 13
        self.b_space_dim = 28
        self.dv_space_dim = 28 + 13 + 13
        self.b_space_workers = b_space_workers
        self.min_a_space_threshold = min_a_space_threshold
        self.opt_type = opt_type
        self.xi_bounds = np.array([[-200, -0.05], [200, 0.05]])

    def set_point_optimum_init(self, init_x):
        self.init_x = init_x

    def load_classifier(self, path):
        tag = "offset"
        if self.use_surrogate:
            self.classifier = pickle.load(open(f"{path}/classifier_{tag}.pkl", "rb"))
            self.scaler = pickle.load(open(f"{path}/scaler_{tag}.pkl", "rb"))
        else:
            raise ValueError("Surrogate not used")

    def fitness(
        self,
        x,
    ):
        if self.use_compensation:
            penalisation_term = 0.0
            x_a_bounds = x
            self.a_space_lb = x_a_bounds[: self.a_space_dim]
            self.a_space_ub = x_a_bounds[self.a_space_dim :]
            bound_diff = self.a_space_ub - self.a_space_lb
            constraint_violation = bound_diff < self.min_a_space_threshold
            if np.any(constraint_violation):
                penalisation_term = 1e3 * np.mean(abs(bound_diff[constraint_violation]))
                return penalisation_term
            assert self.init_x is not None, "Initial point optimum not set"
            init_x_a_space = self.init_x[self.b_space_dim :]
            fixed_offset = np.array([25] * 28 + [40] * 13 + [0.5] * 13)
            a_space_samples = self.parent_rng.uniform(
                init_x_a_space
                + self.a_space_lb * self.search_margin * fixed_offset[28:],
                init_x_a_space
                + self.a_space_ub * self.search_margin * fixed_offset[28:],
                (self.num_a_space_samples, (self.a_space_dim)),
            )
            xi_samples = self.parent_rng.uniform(
                self.xi_bounds[0],
                self.xi_bounds[1],
                (
                    self.num_a_space_samples,
                    len(self.xi_bounds[0]),
                ),
            )
            init_x_b_space = self.init_x[: self.b_space_dim]
            b_space_samples = self.parent_rng.uniform(
                init_x_b_space - fixed_offset[:28],
                init_x_b_space + fixed_offset[:28],
                (self.num_b_space_samples, self.b_space_dim),
            )
            a_xi_combined_samples = np.concatenate(
                (a_space_samples, xi_samples), axis=1
            )
            if self.use_surrogate:
                clf_pred_vals = []
                # CAUTION: TODO: This might break the system if you dont have enough memory to allocate
                # repeated_b = np.tile(b_space_samples, (len(a_xi_combined_samples), 1))
                # repeated_a = np.repeat(
                #     a_xi_combined_samples, self.num_b_space_samples, axis=0
                # )
                # big_combo = np.concatenate((repeated_b, repeated_a), axis=1)
                # dv_xi_scaled = self.scaler.transform(big_combo)
                # probabilities = self.classifier.predict_proba(dv_xi_scaled)[:, 1]
                # pred_matrix = (probabilities > 0.9).reshape(
                #     len(a_xi_combined_samples), self.num_b_space_samples
                # )
                # clf_pred_vals = pred_matrix.any(axis=1).astype(int)
                # mu_omega_x_zeta = np.mean(clf_pred_vals)

                # for a safer execution
                for a_xi_sample in a_xi_combined_samples:
                    dv_xi = np.concatenate(
                        (
                            b_space_samples,
                            np.tile(a_xi_sample, (self.num_b_space_samples, 1)),
                        ),
                        axis=1,
                    )
                    dv_xi_scaled = self.scaler.transform(dv_xi)
                    clf_pred_vals.append(
                        np.any(self.classifier.predict_proba(dv_xi_scaled)[:1] > 0.95)
                    )
                mu_omega_x_zeta = np.mean(clf_pred_vals)

            else:
                raise ValueError("Only with surrogate implemented")
        else:
            x_dv_bounds = x
            self.dv_space_lb = x_dv_bounds[: self.dv_space_dim]
            self.dv_space_ub = x_dv_bounds[self.dv_space_dim :]
            min_dv_space_threshold = 2.2e-2
            min_control_threshold = 0
            bound_diff = self.dv_space_ub - self.dv_space_lb
            if np.logical_or(
                np.any(bound_diff[28:] < min_dv_space_threshold),
                np.any(bound_diff[:28] < min_control_threshold),
            ):
                return 1e4 * np.mean(
                    abs(bound_diff[np.where(bound_diff < min_dv_space_threshold)])
                )

            assert self.init_x is not None, "Initial point optimum not set"
            dv_space_samples = self.parent_rng.uniform(
                (1 + self.dv_space_lb) * self.init_x,
                (1 + self.dv_space_ub) * self.init_x,
                (self.num_a_space_samples, self.dv_space_dim),
            )
            zeta_samples = self.parent_rng.uniform(
                self.zeta_bounds[0],
                self.zeta_bounds[1],
                (self.num_a_space_samples, len(self.zeta_bounds[0])),
            )
            dv_zeta_combined_samples = np.concatenate(
                (dv_space_samples, zeta_samples), axis=1
            )
            if self.use_surrogate:
                raise ValueError("Only without surrogate implemented")
            else:
                out_dats = np.array(
                    Parallel(n_jobs=self.b_space_workers)(
                        delayed(self.compute_fitness_components_with_noise)(
                            dv_zeta_combined_samples[ii]
                        )
                        for ii in range(len(dv_zeta_combined_samples))
                    )
                )
                mu_omega_x_zeta = (
                    np.sum(out_dats[:, 0] >= 20) / self.num_a_space_samples
                )
        return -mu_omega_x_zeta + penalisation_term

    def point_fitness(self, x):
        if self.use_compensation:
            raise ValueError("Only without compensation implemented")
        else:
            xi_samples = self.parent_rng.uniform(
                self.xi_bounds[0],
                self.xi_bounds[1],
                (
                    self.num_a_space_samples,
                    len(self.xi_bounds[0]),
                ),
            )
            dv_xi_combined_samples = np.concatenate(
                (np.tile(x, (self.num_a_space_samples, 1)), xi_samples), axis=1
            )
            if self.use_surrogate:
                dv_xi_scaled = self.scaler.transform(dv_xi_combined_samples)
                clf_pred_vals = self.classifier.predict(dv_xi_scaled)

                mu_omega_pb_x_zeta = np.sum(clf_pred_vals) / self.num_a_space_samples
            else:
                raise ValueError("Only with surrogate implemented")
        return -mu_omega_pb_x_zeta

    def compute_fitness_components_with_noise(self, x):
        (
            Tf,
            tau_mlog,
            tau_flog,
            mv_log,
            motor_constant_matrix,
        ) = run_walking_sim(
            x[:54],
            self.scenario,
            self.parent_rng,
            x[-1],
            x[-2],
            self.opt_mode,
            self.randomise_obstacle,
            self.meshcat,
            time_step=self.time_step,
        )
        Tf, P_f, P_j = self.compute_qois(
            Tf, tau_mlog, tau_flog, mv_log, motor_constant_matrix
        )
        return Tf, P_f, P_j

    def get_bounds(self):
        if self.opt_type == "ssdt":
            if self.use_compensation:
                return np.array(
                    [
                        [-1] * (13 + 13) * 2,
                        [1] * (13 + 13) * 2,
                    ]
                )
            else:
                return np.array(
                    [
                        [-self.search_margin] * (28 + 13 + 13) * 2,
                        [self.search_margin] * (28 + 13 + 13) * 2,
                    ]
                )
        elif self.opt_type == "pb":
            filetag = None
            with open(
                f"../optimisation_logs/gamma/{filetag}_opt_progress.pkl",
                "rb",
            ) as f:
                all_sols = pickle.load(f)

            all_res = np.array([all_sols[k]["res"] for k in all_sols.keys()])
            all_res = all_res.flatten()
            all_x = np.array([all_sols[k]["x"] for k in all_sols.keys()])
            all_x = all_x.reshape(-1, all_x.shape[-1])
            selected_ssdt_optimum = all_x[np.where(all_res == np.min(all_res))[0][-1]]
            super_problem_bounds = np.array(
                [
                    [0.5] * 28 + [50] * 13 + [0.1] * 13,
                    [500] * 28 + [160] * 13 + [1.5] * 13,
                ]
            )

            this_problem_bounds = np.array(
                [
                    list((1 - self.search_margin) * self.init_x[:28])
                    + list((1 + selected_ssdt_optimum[:26]) * self.init_x[28:]),
                    list((1 + self.search_margin) * self.init_x[:28])
                    + list((1 + selected_ssdt_optimum[26:]) * self.init_x[28:]),
                ]
            )
            this_problem_bounds_clipped = np.array(
                list(
                    np.clip(
                        this_problem_bounds[0],
                        super_problem_bounds[0],
                        super_problem_bounds[1],
                    )
                ),
                list(
                    np.clip(
                        this_problem_bounds[1],
                        super_problem_bounds[0],
                        super_problem_bounds[1],
                    )
                ),
            )
            return this_problem_bounds_clipped

    def evaluate_batch_expected_fitness(self, population, workers):
        temp_solutions = []
        if self.opt_type == "ssdt":
            temp_solutions.append(
                Parallel(n_jobs=workers)(
                    delayed(self.fitness)(population[ii])
                    for ii in range(len(population))
                )
            )
        elif self.opt_type == "pb":
            temp_solutions.append(
                Parallel(n_jobs=workers)(
                    delayed(self.point_fitness)(population[ii])
                    for ii in range(len(population))
                )
            )
        temp_solutions = temp_solutions[0]
        solutions = []
        for ii in range(len(population)):
            solutions.append((population[ii], temp_solutions[ii]))
        return solutions

    def evaluate(
        self,
        population,
        generation,
        file_tag,
        use_surrogate=False,
        clf=None,
        scaler=None,
    ):
        solutions = []
        if not use_surrogate:
            for ii in tqdm(range(len(population))):
                temp_solution = self.SSE_fitness(
                    population[ii], ii, generation, file_tag, use_surrogate, clf, scaler
                )
                solutions.append((population[ii], temp_solution))
            return solutions

        if use_surrogate:
            for ii in tqdm(range(len(population))):
                temp_solution = self.SSE_fitness_classifier(population[ii], clf, scaler)
                solutions.append((population[ii], temp_solution))
            return solutions
