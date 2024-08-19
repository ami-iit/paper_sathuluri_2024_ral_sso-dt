# %%
import numpy as np
from include.utils import (
    K_N,
    TSIDParameterTuning,
    sim_joint_order,
    joint_motor_data_dict,
)

# %%

selected_labels = sim_joint_order[:6] + sim_joint_order[12:15] + sim_joint_order[15:19]
selected_idxs = np.concatenate((np.arange(0, 6), np.arange(12, 15), np.arange(15, 19)))

# %% For QDD

tau_h = lambda x: 5.48 * x**0.97

mit_robot_joints = np.array(
    [
        [6, 68, 45],  # hip
        [6, 33.6, 55],  # hip
        [6, 33.6, 55],  # hip
        [12, 136.0, 22.5],  # knee
        [9.33, 52.2, 35],  # ankle
        [np.nan, np.nan, np.nan],  # ankle
        [np.nan, np.nan, np.nan],  # torso
        [np.nan, np.nan, np.nan],  # torso
        [np.nan, np.nan, np.nan],  # torse
        [6, 33.6, 55],  # shoulder-pitch
        [6, 33.6, 55],  # shoulder-pitch
        [np.nan, np.nan, np.nan],  # shoulder-pitch
        [9.33, 52.2, 35],  # elbow
    ]
)

mit_N = mit_robot_joints[:, 0]

x_N_mit_qdd = np.concatenate(
    (
        mit_N[:6],
        mit_N[:6],
        mit_N[6:9],
        mit_N[9:],
        mit_N[9:],
    )
)

mit_m = mit_robot_joints[:, 1] / mit_robot_joints[:, 0]

x_m_mit_qdd = np.concatenate(
    (
        mit_m[:6],
        mit_m[:6],
        mit_m[6:9],
        mit_m[9:],
        mit_m[9:],
    )
)


# %% For HD

gear_ratio_vector = list(np.diag(K_N))
selected_baseline_gears = (
    gear_ratio_vector[:6] + gear_ratio_vector[12:15] + gear_ratio_vector[15:19]
)

selected_baseline_motors = [
    joint_motor_data_dict[selected_labels[ii]]["continuous_stall_torque"]
    for ii in range(len(selected_labels))
]

# %%
# baseline gains
temp_TSID = TSIDParameterTuning()
baseline_gains = temp_TSID.x_k_init

# %%

list_kp = []
list_kd = []
for ii in sim_joint_order[:6] + sim_joint_order[-8:-4]:
    list_kp.append(ii + "_kp")
for ii in sim_joint_order[:6] + sim_joint_order[-8:-4]:
    list_kd.append(ii + "_kd")
gain_list_names = (
    list_kp
    + list_kd
    + ["foot_kp_lin", "foot_kd_lin", "foot_kp_ang", "foot_kd_ang"]
    + ["root_ori_kp", "root_ori_kd"]
    + ["CoM_kp", "CoM_kd"]
)
