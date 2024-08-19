from joblib import Parallel, delayed
import numpy as np
from include.simulation_system import run_walking_sim
from tqdm import tqdm


class GainTuningProblem:
    def __init__(
        self,
        seed,
        scenario,
        add_base_noise,
        add_root_disturbance,
        opt_mode,
        randomise_obstacle,
        robot_type,
        meshcat,
    ):
        self.parent_rng = np.random.default_rng(seed)
        self.scenario = scenario
        self.add_base_noise = add_base_noise
        self.add_root_disturbance = add_root_disturbance
        self.opt_mode = opt_mode
        self.randomise_obstacle = randomise_obstacle
        self.robot_type = robot_type
        self.meshcat = meshcat
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

    def fitness(self, x, child_rng):
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
            self.robot_type,
            self.meshcat,
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
            self.robot_type,
            self.meshcat,
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
            self.robot_type,
            self.meshcat,
            time_step=4e-4,
        )
        Tf, P_f, P_j = self.compute_qois(
            Tf, tau_mlog, tau_flog, mv_log, motor_constant_matrix
        )
        return Tf, P_f, P_j, self.compute_metric(Tf, P_f, P_j, x)

    def expected_fitness(self, x, workers, compute_expectation):
        array_fitness = np.asarray(
            Parallel(n_jobs=workers)(
                delayed(self.fitness)(x) for _ in range(compute_expectation)
            )
        )
        mean_outcome = np.mean(array_fitness, axis=0)
        return mean_outcome

    def evaluate_expected_fitness(self, x, workers, compute_expectation):
        return (x, self.expected_fitness(x, workers, compute_expectation))

    def evaluate_batch_expected_fitness(self, population, workers, compute_expectation):
        temp_solutions = []
        child_rngs = self.parent_rng.spawn(len(population))

        temp_solutions.append(
            Parallel(n_jobs=workers)(
                delayed(self.fitness)(population[ii], child_rngs[ii])
                for ii in range(len(population))
                for _ in range(compute_expectation)
            )
        )
        temp_solutions = temp_solutions[0]
        solutions = []
        for ii in range(len(population)):
            temp_fitness = []
            for jj in range(compute_expectation):
                temp_fitness.append(temp_solutions[ii + jj])
            solutions.append((population[ii], np.mean(temp_fitness)))
        return solutions

    def get_bounds(self):
        match self.opt_mode:
            case "co_design":
                if self.robot_type == "QDD":
                    return (
                        [0] * 28 + [1] * 13 + [1e-3] * 13,
                        [1000] * 28 + [15] * 13 + [2] * 13,
                    )
                if self.robot_type == "harmonic":
                    return (
                        [0] * 28 + [1] * 13 + [1e-3] * 13,
                        [1000] * 28 + [160] * 13 + [2] * 13,
                    )
            case "gain_tuning":
                return ([0] * 28, [1000] * 28)
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


class walking_SSE(GainTuningProblem):
    def __init__(
        self,
        seed=302479294035374746826607037086319450926,
        scenario="None",
        add_base_noise=0,
        add_root_disturbance=0,
        opt_mode="co_design",
        randomise_obstacle=0,
        # robot_type="QDD",
        robot_type="harmonic",
        meshcat=None,
    ):
        super().__init__(
            seed=seed,
            scenario=scenario,
            add_base_noise=add_base_noise,
            add_root_disturbance=add_root_disturbance,
            opt_mode=opt_mode,
            randomise_obstacle=randomise_obstacle,
            robot_type=robot_type,
            meshcat=meshcat,
        )

        self.opt_prob_dim = 28 + 13 + 13
        self.num_DV_samples = 48
        self.workers = 48
        self.compute_expectation = 0
        self.init_x = np.empty((0, 0))
        self.max_fitness = 31000
        self.max_dv_range = 0.15
        self.a_space_size_weight = 1
        self.b_space_size_weight = 1

    def fitness_and_norm(self, x, child_rng):
        x_k = x[:-2]
        base_noise_sample = x[-2]
        root_disturbance_sample = x[-1]
        Tf, P_f, P_j, metric = self.compute_fitness_and_components_with_noise(
            x_k, base_noise_sample, root_disturbance_sample, child_rng
        )
        return (metric / self.max_fitness), Tf, P_f, P_j

    def SSE_fitness(
        self,
        x,
        pop_num,
        generation,
        file_tag=None,
        use_surrogate=False,
        clf=None,
        scaler=None,
    ):

        x_a_bounds = x

        # volume of the A space
        self.dv_lb = np.asarray(
            [x_a_bounds[ii] for ii in range(0, (self.opt_prob_dim - 28) * 2, 2)]
        )
        self.dv_ub = np.asarray(
            [x_a_bounds[ii] for ii in range(1, (self.opt_prob_dim - 28) * 2, 2)]
        )
        DV_bounds_range = (self.dv_ub - self.dv_lb) / self.max_dv_range
        mu_omega_x = np.sum(DV_bounds_range) / (
            self.opt_prob_dim - 28
        )  # max value is 1

        # volume of B space
        init_x_DVs = self.init_x[28:]
        sample_DVs = self.parent_rng.uniform(
            (1 + self.dv_lb) * init_x_DVs,
            (1 + self.dv_ub) * init_x_DVs,
            (self.num_DV_samples, (self.opt_prob_dim - 28)),
        )
        # sample from B space
        zeta_samples = np.random.uniform(
            self.zeta_bounds[0],
            self.zeta_bounds[1],
            (self.num_DV_samples, len(self.zeta_bounds[0])),
        )
        # pre-sampled x_k
        b_space_sample_range = 0.5
        b_space_samples = 50
        x_k_init = self.init_x[:28]
        self.parent_rng.uniform(
            (1 + b_space_sample_range) * x_k_init,
            (1 - b_space_sample_range) * x_k_init,
            (b_space_samples, 28),
        )

        combined_samples = np.concatenate((sample_DVs, zeta_samples), axis=1)

        if not use_surrogate:
            child_rngs = self.parent_rng.spawn(self.num_DV_samples)
            outs = np.asarray(
                Parallel(n_jobs=int(self.workers))(
                    delayed(self.fitness_and_norm)(combined_samples[ii], child_rngs[ii])
                    for ii in tqdm(range(self.num_DV_samples))
                )
            )
            metric_values = outs[:, 0]
            mean_metric = np.mean(metric_values)
            mu_omega_zeta = 1 - mean_metric  # max value is also 1

            out_data = outs
            filename = "../optimisation_logs/{}_{}_{}_SSE/{}_{}.pkl".format(
                file_tag,
                self.opt_mode,
                self.robot_type,
                generation,
                pop_num,
            )
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, "wb") as f:
                temp_dict = {}
                temp_dict["x"] = combined_samples
                temp_dict["out_dat"] = out_data
                pickle.dump(temp_dict, f)
        else:
            combined_samples_scaled = scaler.transform(combined_samples)
            metric_values = clf.predict(combined_samples_scaled)
            mean_metric = np.mean(metric_values)
            mu_omega_zeta = 1 - mean_metric

        V = (self.a_space_size_weight * mu_omega_zeta) * (
            self.b_space_size_weight * mu_omega_x
        )
        mu_omega_zeta_c = 0.7
        if mu_omega_zeta < mu_omega_zeta_c:
            penal = 1e3 * (mu_omega_zeta_c - mu_omega_zeta)
            return -V + penal
        else:
            return -V

    def get_bounds(self):
        self.zeta_bounds = np.array([[-0.1, -20000], [0.1, 20000]])
        return np.array(
            [
                [-self.max_dv_range / 2, 0] * (self.opt_prob_dim - 28),
                [0, self.max_dv_range / 2] * (self.opt_prob_dim - 28),
            ]
        )

    def SSE_fitness_classifier(
        self,
        x,
        clf=None,
        scaler=None,
    ):
        x_a_bounds = x
        # volume of the A space
        self.dv_lb = np.asarray([x_a_bounds[ii] for ii in range(0, 26 * 2, 2)])
        self.dv_ub = np.asarray([x_a_bounds[ii] for ii in range(1, 26 * 2, 2)])
        DV_bounds_range = (self.dv_ub - self.dv_lb) / self.max_dv_range
        mu_omega_x = np.sum(DV_bounds_range) / (26)  # max value is 1

        # volume of B space
        init_x_DVs = self.init_x[28:]
        sample_DVs = self.parent_rng.uniform(
            (1 + self.dv_lb) * init_x_DVs,
            (1 + self.dv_ub) * init_x_DVs,
            (self.num_DV_samples, 26),
        )
        # sample from B space
        zeta_samples = np.random.uniform(
            self.zeta_bounds[0],
            self.zeta_bounds[1],
            (self.num_DV_samples, len(self.zeta_bounds[0])),
        )
        ab_combined_samples = np.concatenate((sample_DVs, zeta_samples), axis=1)

        # pre-sampled x_k (the compensation space)
        comp_space_sample_range = 0.01
        comp_space_num_samples = 50
        x_k_init = self.init_x[:28]
        comp_space_samples = self.parent_rng.uniform(
            (1 - comp_space_sample_range) * x_k_init,
            (1 + comp_space_sample_range) * x_k_init,
            (comp_space_num_samples, 28),
        )

        clf_pred_vals = []
        for ab_sample in ab_combined_samples:
            full_sample = np.concatenate(
                (comp_space_samples, np.tile(ab_sample, (comp_space_num_samples, 1))),
                axis=1,
            )

            full_sample_scaled = scaler.transform(full_sample)
            clf_pred_vals.append(clf.predict(full_sample_scaled).any())

        mc_integ = np.mean(clf_pred_vals) / self.num_DV_samples
        mu_omega_zeta = mc_integ  # max value is 1

        V = (self.a_space_size_weight * mu_omega_zeta) * (
            self.b_space_size_weight * mu_omega_x
        )
        mu_omega_zeta_c = 0
        if mu_omega_zeta < mu_omega_zeta_c:
            penal = 1e3 * (mu_omega_zeta_c - mu_omega_zeta)
            return -V + penal
        else:
            return -V

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
