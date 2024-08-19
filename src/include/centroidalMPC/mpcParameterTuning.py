import numpy as np
from datetime import timedelta


class MPCParameterTuning:

    def __init__(self) -> None:
        self.com_weight = np.asarray([100, 100, 1000])
        self.contact_position_weight = 1e3
        self.force_rate_change_weight = np.asarray([10.0, 10.0, 10.0])
        self.angular_momentum_weight = 1e5
        self.contact_force_symmetry_weight = 1.0

        self.contact_position_weight = np.array(3)
        self.fore_rate_change_weight = [1.0]
        self.angular_momentum_weight = np.array(3)
        self.time_horizon = timedelta(seconds=1.2)
        self.time_step_planner = 0.1
