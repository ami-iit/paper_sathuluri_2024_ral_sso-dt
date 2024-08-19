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
