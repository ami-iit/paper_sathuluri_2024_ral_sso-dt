# %%
import glob
import logging
import time
import xml.etree.ElementTree as ET
from copy import copy as pycopy
from datetime import timedelta
from IPython import get_ipython
import os
from pathlib import Path

import idyntree.bindings as idyn
import matplotlib.pyplot as plt

# %%
import numpy as np
import resolve_robotics_uri_py
from adam.casadi.computations import KinDynComputations
from pydrake.geometry import (
    AddCompliantHydroelasticProperties,
    AddContactMaterial,
    AddRigidHydroelasticProperties,
    Box,
    HalfSpace,
    MeshcatVisualizer,
    ProximityProperties,
    Sphere,
)
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.systems.framework import DiagramBuilder
import logging

logging.basicConfig(level=logging.CRITICAL)


# %%
def find_running_as_notebook():
    return (
        "COLAB_TESTING" not in os.environ
        and get_ipython()
        and hasattr(get_ipython(), "kernel")
    )


# %%

s_init = {
    "torso_pitch": 0,
    "torso_roll": 0,
    "torso_yaw": 0,
    "r_hip_pitch": 1.0,
    "r_hip_roll": 0.01903913,
    "r_hip_yaw": -0.0172335,
    "r_knee": -1.2220763,
    "r_ankle_pitch": -0.52832664,
    "r_ankle_roll": -0.02720832,
    "l_hip_pitch": 1,
    "l_hip_roll": 0.0327311,
    "l_hip_yaw": -0.02791293,
    "l_knee": -1.22200495,
    "l_ankle_pitch": -0.52812215,
    "l_ankle_roll": -0.04145696,
    "r_shoulder_pitch": 0.02749586,
    "r_shoulder_roll": 0.25187149,
    "r_shoulder_yaw": -0.14300417,
    "r_elbow": 0.6168618,
    "l_shoulder_pitch": 0.03145343,
    "l_shoulder_roll": 0.25644825,
    "l_shoulder_yaw": -0.14427671,
    "l_elbow": 0.61634549,
}

con_joint_order = [
    "r_hip_pitch",  # 0
    "r_hip_roll",  # 1
    "r_hip_yaw",  # 2
    "r_knee",  # 3
    "r_ankle_pitch",  # 4
    "r_ankle_roll",  # 5
    "l_hip_pitch",  # 6
    "l_hip_roll",  # 7
    "l_hip_yaw",  # 8
    "l_knee",  # 10
    "l_ankle_pitch",  # 11
    "l_ankle_roll",  # 12
    "r_shoulder_pitch",  # 13
    "r_shoulder_roll",  # 14
    "r_shoulder_yaw",  # 15
    "r_elbow",  # 16
    "l_shoulder_pitch",  # 17
    "l_shoulder_roll",  # 18
    "l_shoulder_yaw",  # 19
    "l_elbow",  # 20
]

# use foot size for simulation
# use a conservative value in control computation
xMinMax = [-0.1, 0.1]
yMinMax = [-0.05, 0.05]

# for ergoCub
urdf_path_sim = "../urdfs/model_ergoCub_foot_no_collisions.urdf"
urdf_path_con = resolve_robotics_uri_py.resolve_robotics_uri(
    "package://ergoCub/robots/ergoCubSN000/model.urdf"
)
mesh_path = urdf_path_con.__str__().split("robots/ergoCubSN000/model.urdf")[0]
urdf_path_viz = urdf_path_con


# %%
def get_temp_plant():
    builder = DiagramBuilder()
    time_step = 0.0
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant, scene_graph)
    parser.package_map().Add("ergoCub", mesh_path)
    robot_model_sim = parser.AddModels(urdf_path_sim)[0]
    plant.Finalize()
    return plant, robot_model_sim


plant, robot_model_sim = get_temp_plant()


def get_sim_joint_order(plant, robot_model_sim):
    sim_joint_order = []
    for ii in plant.GetJointIndices(robot_model_sim):
        jj = plant.get_joint(ii)
        if jj.type_name() == "revolute":
            sim_joint_order.append(jj.name())
    return sim_joint_order


sim_joint_order = get_sim_joint_order(plant, robot_model_sim)


def get_mapping_matrices(sim_joint_order, con_joint_order):
    to_lock = []

    sim2conMap = np.zeros((len(con_joint_order), len(sim_joint_order)))
    for ii in range(len(con_joint_order)):
        try:
            sim2conMap[ii, sim_joint_order.index(con_joint_order[ii])] = 1.0
        except ValueError:
            logging.info(
                "Additional data not used in sim is ignored: {}".format(
                    con_joint_order[ii]
                )
            )
            pass

    con2simMap = np.zeros((len(sim_joint_order), len(con_joint_order)))
    for ii in range(len(sim_joint_order)):
        try:
            con2simMap[ii, con_joint_order.index(sim_joint_order[ii])] = 1.0
        except ValueError:
            logging.info(
                "Tag not found, locking joint in sim: {}".format(sim_joint_order[ii])
            )
            to_lock.append(sim_joint_order[ii])
    return sim2conMap, con2simMap, to_lock


sim2conMap, con2simMap, to_lock = get_mapping_matrices(sim_joint_order, con_joint_order)


def qRemap(qval, qFrom):
    if qFrom == "sim":
        return sim2conMap.dot(qval)
    if qFrom == "con":
        return con2simMap.dot(qval)


# test for the mapping
temp_con_list = []
for ii in range(len(con_joint_order)):
    temp_con_list.append(s_init[con_joint_order[ii]])
temp_con_list = np.array(temp_con_list)

temp_sim_list = []
for ii in range(len(sim_joint_order)):
    temp_sim_list.append(s_init[sim_joint_order[ii]])
temp_sim_list = np.array(temp_sim_list)

s_init_sim = pycopy(temp_sim_list)
s_init_con = qRemap(temp_sim_list, "sim")
res = s_init_con - temp_con_list
logging.debug(res)


def compute_info(meshcat=None):
    builder = DiagramBuilder()
    time_step = 0.0
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant, scene_graph)
    parser.package_map().Add("ergoCub", mesh_path)
    robot_model_sim = parser.AddModels(urdf_path_sim)[0]

    plant.Finalize()
    if meshcat != None:
        visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    plant.SetPositions(
        plant_context,
        np.concatenate((np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3), s_init_sim)),
    )
    l_sole_T = plant.CalcRelativeTransform(
        plant_context,
        plant.GetBodyByName("l_sole").body_frame(),
        plant.GetBodyByName("root_link").body_frame(),
    )
    base_ori_init_wxyz = l_sole_T.rotation().ToQuaternion().wxyz()
    base_pos_init = l_sole_T.translation()

    plant.SetPositions(
        plant_context,
        np.concatenate((base_ori_init_wxyz, base_pos_init, temp_sim_list)),
    )
    diagram.ForcedPublish(diagram_context)

    root_link_T = plant.CalcRelativeTransform(
        plant_context,
        plant.world_frame(),
        plant.GetBodyByName("root_link").body_frame(),
    )
    root_link_ori_wxyz = root_link_T.rotation().ToQuaternion().wxyz()

    com_init = plant.CalcCenterOfMassPositionInWorld(plant_context)
    robot_mass = plant.CalcTotalMass(plant_context)
    l_sole_T = (
        plant.get_body(plant.GetBodyByName("l_sole").index())
        .body_frame()
        .CalcPoseInWorld(plant_context)
    )
    r_sole_T = (
        plant.get_body(plant.GetBodyByName("r_sole").index())
        .body_frame()
        .CalcPoseInWorld(plant_context)
    )

    return (
        base_ori_init_wxyz,
        base_pos_init,
        com_init,
        robot_mass,
        root_link_ori_wxyz,
        l_sole_T,
        r_sole_T,
    )


(
    base_ori_init_wxyz,
    base_pos_init,
    com_init,
    robot_mass,
    root_link_ori_wxyz,
    l_sole_T,
    r_sole_T,
) = compute_info()


# %%
def plot_data(log, tfinal, name=None, index=None):
    base_labels = ["qw", "qx", "qy", "qz", "px", "py", "pz"]
    pos_labels = base_labels + sim_joint_order
    sample_times = log.sample_times()
    findex = np.where(sample_times == tfinal)[0][0]
    logdata = log.data()
    if name == "q":
        qdata = logdata[: plant.num_positions()]
        if index == None:
            index = range(qdata.shape[0])
        for ii in index:
            plt.plot(
                sample_times[:findex],
                qdata[ii][:findex],
                label=pos_labels[ii] + " " + str(ii),
            )
        plt.title("Joint positions vs time")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
        plt.show()

    if name == "v":
        vdata = logdata[plant.num_positions() :]
        if index == None:
            index = range(vdata.shape[0])
        for ii in index:
            plt.plot(
                sample_times[:findex],
                vdata[ii][:findex],
                label=pos_labels[ii] + " " + str(ii),
            )
        plt.title("Joint velocities vs time")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()

    if name == "u":
        pos_labels = sim_joint_order
        taudata = logdata[: plant.num_actuators()]
        if index == None:
            index = range(taudata.shape[0])
        for ii in index:
            plt.plot(
                sample_times[:findex],
                taudata[ii][:findex],
                label=pos_labels[ii] + " " + str(ii),
            )
        plt.title("Joint torques vs time")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()

    if name == None:
        for ii in range(logdata.shape[0]):
            plt.plot(
                sample_times[:findex],
                logdata[ii][:findex],
                label="index_" + str(ii),
            )


# %%
class RobotModel(KinDynComputations):
    def __init__(self, urdfstring: str, urdf_path: str) -> None:
        self.urdf_string = urdfstring
        self.robot_name = "ergoCubSim"
        self.joint_name_list = con_joint_order
        self.base_link = "root_link"
        self.left_foot_frame = "l_sole"
        self.right_foot_frame = "r_sole"
        self.gravity = idyn.Vector3()
        self.gravity.zero()
        self.gravity.setVal(2, -9.81)
        self.H_b = idyn.Transform()
        super().__init__(urdf_path, self.joint_name_list, self.base_link)

    def get_idyntree_kyndyn(self):
        self.model_loader = idyn.ModelLoader()
        self.model_loader.loadReducedModelFromString(
            self.urdf_string.decode("utf-8"), self.joint_name_list
        )
        self.model = self.model_loader.model()
        kindyn = idyn.KinDynComputations()
        kindyn.loadRobotModel(self.model)
        return kindyn


# %%
class TSIDParameterTuning:
    def __init__(self) -> None:
        self.CoM_Kp = 15.0
        self.CoM_Kd = 7.0
        # ---------------------------------------------
        # symmetric postural gains
        leg = np.array([130, 80, 20, 60]) * 2.8
        arm = np.array([150, 20, 20, 180, 140, 20]) * 2.8
        self.postural_Kp = np.concatenate((leg, leg, arm, arm))

        leg_kd = np.power(leg, 1 / 2) / 10
        arm_kd = np.power(arm, 1 / 2) / 10
        self.postural_Kd = np.concatenate((leg_kd, leg_kd, arm_kd, arm_kd))
        # ---------------------------------------------
        # foot tracking gains
        self.foot_tracking_task_kp_lin = 100.0
        self.foot_tracking_task_kd_lin = 7.0
        self.foot_tracking_task_kp_ang = 300.0
        self.foot_tracking_task_kd_ang = 10.0
        # ---------------------------------------------
        # root link tracking gains
        self.root_link_kp_ang = 20.0
        self.root_link_kd_ang = 10.0
        # ---------------------------------------------
        # task weights
        self.postural_weight = 1e1 * np.ones(len(self.postural_Kp))
        self.root_tracking_task_weight = 1e1 * np.ones(3)
        self.x_k_init = np.concatenate(
            (
                arm,
                leg,
                arm_kd,
                leg_kd,
                np.asarray(
                    [
                        self.foot_tracking_task_kp_lin,
                        self.foot_tracking_task_kd_lin,
                        self.foot_tracking_task_kp_ang,
                        self.foot_tracking_task_kd_ang,
                        self.root_link_kp_ang,
                        self.root_link_kd_ang,
                        self.CoM_Kp,
                        self.CoM_Kd,
                    ]
                ),
            )
        )

    def set_postural_gain(self, arm, leg, arm_kd, leg_kd):
        self.postural_Kp = np.concatenate([leg, leg, arm, arm])
        self.postural_kd = np.concatenate([leg_kd, leg_kd, arm_kd, arm_kd])

    def set_foot_task(self, kp_lin, kd_lin, kp_ang, kd_ang):
        self.foot_tracking_task_kp_lin = kp_lin
        self.foot_tracking_task_kd_lin = kd_lin
        self.foot_tracking_task_kp_ang = kp_ang
        self.foot_tracking_task_kd_ang = kd_ang

    def set_root_task(self, kp_ang, kd_ang):
        self.root_link_kp_ang = kp_ang
        self.root_link_kd_ang = kd_ang

    def set_weights(self, postural_weight, root_weight):
        self.postural_weight = postural_weight
        self.root_tracking_task_weight = root_weight

    def set_com_task(self, kp_com, kd_com):
        self.CoM_Kd = kd_com
        self.CoM_Kp = kp_com

    def set_from_x_k(self, x_k):
        # arm, leg, arm_kd, leg_kd
        self.set_postural_gain(x_k[:4], x_k[4:10], x_k[10:14], x_k[14:20])
        self.set_foot_task(x_k[20], x_k[21], x_k[22], x_k[23])
        self.set_root_task(x_k[24], x_k[25])
        self.set_com_task(x_k[26], x_k[27])


# %%
class MPCParameterTuning:
    def __init__(self) -> None:
        self.com_weight = np.asarray([10, 10, 500])
        self.contact_position_weight = np.asarray([1e3]) / 1e1
        self.force_rate_change_weight = np.asarray([10.0, 10.0, 10.0]) / 1e2
        self.angular_momentum_weight = np.asarray([1e5]) / 1e3
        self.contact_force_symmetry_weight = np.asarray([1.0]) / 1e2
        self.time_horizon = timedelta(seconds=1.2)

    def set_parameters(
        self,
        com_weight,
        contac_position,
        force_rate_change,
        angular_mom_weight,
        contact_force_symmetry_weight,
    ):
        self.com_weight = com_weight
        self.contact_position_weight = contac_position
        self.fore_rate_change_weight = force_rate_change
        self.angular_momentum_weight = angular_mom_weight
        self.contact_force_symmetry_weight = contact_force_symmetry_weight

    def set_from_xk(self, x_k):
        self.set_parameters(x_k[:3], x_k[3], x_k[4:7], x_k[7], x_k[8])


# %%


def add_ground_with_friction(plant):
    surface_friction_ground = CoulombFriction(static_friction=1.0, dynamic_friction=1.0)
    proximity_properties_ground = ProximityProperties()
    AddContactMaterial(1e4, 1e7, surface_friction_ground, proximity_properties_ground)
    AddRigidHydroelasticProperties(0.01, proximity_properties_ground)

    plant.RegisterCollisionGeometry(
        plant.world_body(),
        RigidTransform(),
        HalfSpace(),
        "ground_collision",
        proximity_properties_ground,
    )


def add_soft_feet_collisions(plant, xMinMax, yMinMax):
    surface_friction_feet = CoulombFriction(static_friction=1.0, dynamic_friction=1.0)
    proximity_properties_feet = ProximityProperties()
    AddContactMaterial(1e4, 1e7, surface_friction_feet, proximity_properties_feet)
    AddCompliantHydroelasticProperties(0.01, 1e8, proximity_properties_feet)
    for ii in ["l_foot_front", "l_foot_rear", "r_foot_front", "r_foot_rear"]:
        plant.RegisterCollisionGeometry(
            plant.GetBodyByName(ii),
            RigidTransform(np.array([0, 0, -2e-2])),
            Box((xMinMax[1] - xMinMax[0]) / 2, yMinMax[1] - yMinMax[0], 2e-3),
            ii + "_collision",
            proximity_properties_feet,
        )

        plant.RegisterVisualGeometry(
            plant.GetBodyByName(ii),
            RigidTransform(np.array([0, 0, -2e-2])),
            Box((xMinMax[1] - xMinMax[0]) / 2, yMinMax[1] - yMinMax[0], 2e-3),
            ii + "_collision",
            np.array([1.0, 1.0, 1.0, 1]),
        )


def add_soft_arm_collisions(plant):
    surface_friction_feet = CoulombFriction(static_friction=1.0, dynamic_friction=1.0)
    proximity_properties_feet = ProximityProperties()
    AddContactMaterial(1e4, 1e7, surface_friction_feet, proximity_properties_feet)
    AddCompliantHydroelasticProperties(0.01, 1e8, proximity_properties_feet)
    radius = 0.03

    for ii in ["l_forearm", "l_hand_palm", "l_upperarm"]:
        plant.RegisterCollisionGeometry(
            plant.GetBodyByName(ii),
            RigidTransform(),
            Sphere(radius=radius),
            ii + "_collision",
            proximity_properties_feet,
        )
        plant.RegisterVisualGeometry(
            plant.GetBodyByName(ii),
            RigidTransform(),
            Sphere(radius=radius),
            ii + "_collision",
            np.array([1.0, 1.0, 1.0, 1]),
        )


def add_soft_obstacle(
    plant,
    location: RigidTransform,
    obstacle,
    name: str,
):
    surface_friction_feet = CoulombFriction(static_friction=1.0, dynamic_friction=1.0)
    proximity_properties_feet = ProximityProperties()
    AddContactMaterial(1e4, 1e7, surface_friction_feet, proximity_properties_feet)
    AddCompliantHydroelasticProperties(0.01, 1e8, proximity_properties_feet)
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        location,
        obstacle,
        name + "_collision",
        proximity_properties_feet,
    )
    plant.RegisterVisualGeometry(
        plant.world_body(),
        location,
        obstacle,
        name + "_collision",
        np.array([1.0, 1.0, 1.0, 1]),
    )


# %%


def extract_motor_info(params_folder):
    config_files = glob.glob(params_folder + "/*.xml")
    motor_data = {}
    for config_file in config_files:
        tree = ET.parse(config_file)
        root = tree.getroot()
        for child in root:
            if child.attrib["name"] == "GENERAL":
                general_group = child
                for param in general_group:
                    if param.attrib["name"] == "AxisName":
                        joints = param.text.split()
                        joints = [joints[i][1:-1] for i in range(len(joints))]
                    if param.attrib["name"] == "Gearbox_M2J":
                        gear_ratios = param.text.split()
                        gear_ratios = [
                            float(gear_ratios[i]) for i in range(len(gear_ratios))
                        ]
                    if param.attrib["name"] == "MotorType":
                        motor_types = param.text.split()
                        motor_types = [
                            motor_types[i][1:-1] for i in range(len(motor_types))
                        ]
            if child.attrib["name"] == "LIMITS":
                limits_group = child
                for param in limits_group:
                    if param.attrib["name"] == "hardwareJntPosMax":
                        jnt_max = param.text.split()
                        jnt_max = [float(jnt_max[i]) for i in range(len(jnt_max))]
                    if param.attrib["name"] == "hardwareJntPosMin":
                        jnt_min = param.text.split()
                        jnt_min = [float(jnt_min[i]) for i in range(len(jnt_min))]
        dat_list = [gear_ratios, motor_types, jnt_max, jnt_min]
        motor_data.update(dict(zip(joints, list(map(list, zip(*dat_list))))))

    motor_datasheet = {
        "2900524": [0.0585 * 1e-4, 0.025, 0.09, 0.3, 0.89, 6000, 59],
        "2900525": [0.0827 * 1e-4, 0.047, 0.18, 0.8, 0.98, 6000, 110],
        "2900575": [0.0585 * 1e-4, 0.025, 0.09, 0.3, 0.89, 6000, 59],
        "2900576": [0.0827 * 1e-4, 0.047, 0.18, 0.8, 0.98, 6000, 110],
        "2900580": [0.2348 * 1e-4, 0.111, 0.43, 1.5, 0.46, 4000, 179],
        "2900583": [0.0585 * 1e-4, 0.025, 0.09, 0.3, 0.89, 6000, 59],
        "2900584": [0.0827 * 1e-4, 0.047, 0.18, 0.8, 0.98, 6000, 110],
    }
    motor_names = list(motor_datasheet.keys())
    motor_datasheet = dict(
        zip(list(motor_datasheet.keys()), list(motor_datasheet.values()))
    )
    joint_motor_data = dict(
        filter(lambda item: item[0] in sim_joint_order, motor_data.items())
    )
    motor_in_keys = list(motor_datasheet.keys())
    joint_motor_data_dict = {}
    for ii in list(joint_motor_data.keys()):
        idx = motor_in_keys.index(((joint_motor_data[ii])[1]).split("C")[1])
        joint_motor_data[ii] += motor_datasheet[motor_in_keys[idx]]
        motor_data_dict = {
            ii: dict(
                zip(
                    [
                        "gear_ratio",
                        "motor_type",
                        "jnt_max",
                        "jnt_min",
                        "motor_inertia",
                        "torque_constant",
                        "rated_torque",
                        "continuous_stall_torque",
                        "mass",
                        "rated_speed",
                        "rated_power",
                    ],
                    joint_motor_data[ii],
                )
            )
        }
        joint_motor_data_dict.update(motor_data_dict)

    return joint_motor_data_dict, motor_names


params_folder = (
    Path("../../../robots-configuration/ergoCubSN000/hardware/mechanicals/")
    .resolve()
    .__str__()
)
print("Searching for robots-configuration repo in: ", params_folder)
joint_motor_data_dict, motor_names = extract_motor_info(params_folder)

I_m_mat = np.eye(len(sim_joint_order))
for ii in range(len(sim_joint_order)):
    I_m_mat[ii, ii] = joint_motor_data_dict[sim_joint_order[ii]]["motor_inertia"]

# %%

joint_friction_coeffs = dict(zip(sim_joint_order, np.zeros((len(sim_joint_order), 2))))
joint_friction_coeffs["r_hip_pitch"] = [1, 1, 2, 2]
joint_friction_coeffs["r_hip_roll"] = [-1.5, -1.5, -2.8, -2.8]
joint_friction_coeffs["r_hip_yaw"] = [0.2, 0.2, 4.2, 4.2]
joint_friction_coeffs["r_knee"] = [6, 7, 5, 8]
joint_friction_coeffs["r_ankle_pitch"] = [0, 0, 11, 11]
joint_friction_coeffs["r_ankle_roll"] = [0.7, 0.7, 1.3, 1.3]

# Sorrentino, Ines, Giulio Romualdi, and Daniele Pucci. "UKF-Based Sensor Fusion for
# Joint-Torque Sensorless Humanoid Robots." arXiv preprint arXiv:2402.18380 (2024).
friction_coeffs_paper = {
    "r_hip_pitch": {"k0": 4.9, "k1": 4.0, "k2": 0.6},
    "r_hip_roll": {"k0": 4.0, "k1": 4.7, "k2": 0.3},
    "r_hip_yaw": {"k0": 2.5, "k1": 2.6, "k2": 0.5},
    "r_knee": {"k0": 2.3, "k1": 2.7, "k2": 0.1},
    "r_ankle_pitch": {"k0": 2.3, "k1": 2.3, "k2": 0.3},
    "r_ankle_roll": {"k0": 1.3, "k1": 2.0, "k2": 0.3},
}


# %%
# generate gear ratio matrix
def gear_ratio_matrix_fun(joint_motor_dict_data, sim_joint_order):
    K_N = np.eye(len(sim_joint_order))
    for ii in range(len(sim_joint_order)):
        K_N[ii, ii] = joint_motor_dict_data[sim_joint_order[ii]]["gear_ratio"]
    return K_N


K_N = gear_ratio_matrix_fun(joint_motor_data_dict, sim_joint_order)


# %%
def motor_torque_constant_matrix(joint_motor_dict_data, sim_joint_order):
    K_T = np.eye(len(sim_joint_order))
    for ii in range(len(sim_joint_order)):
        K_T[ii, ii] = joint_motor_dict_data[sim_joint_order[ii]]["torque_constant"]
    return K_T


K_T = motor_torque_constant_matrix(joint_motor_data_dict, sim_joint_order)

# %%

f_K0 = np.zeros(len(sim_joint_order))
paper_k0 = np.array([4.9, 4.0, 2.5, 2.3, 2.3, 1.3])
f_K0[[0, 1, 2, 3, 4, 5]] = paper_k0
f_K0[[6, 7, 8, 9, 10, 11]] = paper_k0
f_K0 = np.diag(f_K0)
f_K1 = np.zeros(len(sim_joint_order))
paper_k1 = np.array([4.0, 4.7, 2.6, 2.7, 2.3, 2.0])
f_K1[[0, 1, 2, 3, 4, 5]] = paper_k1
f_K1[[6, 7, 8, 9, 10, 11]] = paper_k1
f_K1 = np.diag(f_K1)
f_K2 = np.zeros(len(sim_joint_order))
paper_k2 = np.array([0.6, 0.3, 0.5, 0.1, 0.3, 0.3])
f_K2[[0, 1, 2, 3, 4, 5]] = paper_k2
f_K2[[6, 7, 8, 9, 10, 11]] = paper_k2
f_K2 = np.diag(f_K2)
# %%
