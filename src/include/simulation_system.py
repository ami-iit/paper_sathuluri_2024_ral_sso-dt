import xml.etree.ElementTree as ET
import numpy as np
import logging

import pydrake.multibody.plant as pmp
import pydrake.systems.primitives as psp
from pydrake.geometry import Box, MeshcatVisualizer, Sphere
from pydrake.math import RigidTransform
from pydrake.multibody.meshcat import ContactVisualizer, ContactVisualizerParams
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import ContactModel
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, EventStatus
from pydrake.systems.primitives import Demultiplexer, ZeroOrderHold
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz

import matplotlib.pyplot as plt

from include.addon_systems import DisturbanceSystem, JointActuation, NoiseAdder
from include.centroidalMPC.centroidalMPC_system import CentroidalMPC
from include.qptsid_system import QPfloatingTSID
from include.utils import (
    RobotModel,
    TSIDParameterTuning,
    MPCParameterTuning,
    add_ground_with_friction,
    add_soft_feet_collisions,
    add_soft_obstacle,
    add_soft_arm_collisions,
    qRemap,
    find_running_as_notebook,
    K_N,
    I_m_mat,
    urdf_path_con,
    mesh_path,
    urdf_path_sim,
    to_lock,
    f_K0,
    f_K1,
    f_K2,
    base_ori_init_wxyz,
    base_pos_init,
    s_init_sim,
    com_init,
    root_link_ori_wxyz,
    xMinMax,
    yMinMax,
)

if find_running_as_notebook():
    from include.utils import plot_data

logging.getLogger().setLevel(logging.CRITICAL)


def run_walking_sim(
    x_dv,
    scenario,
    rng,
    add_base_noise,
    add_root_disturbance,
    opt_mode,
    randomise_obstacle,
    robot_type,
    meshcat,
    time_step=5e-4,
):
    match opt_mode:
        # gets x_k, x_N, x_m/x_tau, gear_ratio_matrix, motor_constant_matrix, motor_inertia_matrix, tau_sat
        case "co_design":
            # only for TSID: arm_kp (4), leg_kp (6), arm_kd (4), leg_kd (6) = 20, foot_task (4), root_task (2), com_task (2) = 28
            x_k = x_dv[:28]
            x_N_s = x_dv[28 : 28 + 13]
            # r_leg (6), l_leg (6), torso (3), r_arm (4), l_arm (4)
            x_N = np.concatenate(
                (x_N_s[:6], x_N_s[:6], x_N_s[6:9], x_N_s[9:], x_N_s[9:])
            )
            x_m_s = x_dv[28 + 13 :]  # mass/torques sampled
            x_m = np.concatenate(
                (x_m_s[:6], x_m_s[:6], x_m_s[6:9], x_m_s[9:], x_m_s[9:])
            )
            gear_ratio_matrix = np.diag(x_N)
            if robot_type == "QDD":
                tau_h = lambda x: 5.48 * x**0.97
                Km_h = lambda x: 0.15 * x**1.39
                Im_h = lambda x: 7.19e-4 * x**1.67
                tau_sat = tau_h(x_m) * x_N
                motor_constant_data = Km_h(x_m)
                motor_constant_matrix = np.diag(1 / motor_constant_data**2)
                motor_inertia_matrix = np.diag(Im_h(x_m))
            if robot_type == "harmonic":
                motor_inertia_matrix = I_m_mat
                tau_sat = x_m * x_N
                motor_constant_data = np.ones(23) * 0.15
                motor_constant_matrix = np.diag(1 / motor_constant_data**2)

        case "gain_tuning":
            x_k = x_dv
            tau_sat = np.ones(23) * 150
            gear_ratio_matrix = K_N
            motor_inertia_matrix = I_m_mat
            motor_constant_data = np.ones(23) * 0.15
            motor_constant_matrix = np.diag(1 / motor_constant_data**2)

        case _:
            return 0

    # -----------------------------------------
    tree = ET.parse(urdf_path_con)  # prevent passing SwigObject object
    root = tree.getroot()
    robot_urdf_string = ET.tostring(root)
    robot_model_con = RobotModel(robot_urdf_string, urdf_path_con)

    builder = DiagramBuilder()
    logging.info("Updating simulation at: {} Hz".format(1 / time_step))
    plant, scene_graph = pmp.AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant, scene_graph)
    parser.package_map().Add("ergoCub", mesh_path)  # for ergoCub
    robot_model_sim = parser.AddModels(urdf_path_sim)[0]
    # -----------------------------------------
    plant.set_contact_model(ContactModel.kHydroelasticsOnly)  # Configure contact
    add_ground_with_friction(plant)
    add_soft_feet_collisions(plant, xMinMax, yMinMax)
    # -----------------------------------------
    if scenario == "rubble" or scenario == "both":
        name = "sphere"
        location = RigidTransform(np.array([0.15, 0, 0]))
        if randomise_obstacle:
            location = RigidTransform(
                np.array(
                    [
                        rng.uniform(0.15, 0.5),
                        rng.uniform(-0.05, 0.05),
                        0,
                    ]
                )
            )
        mesh = Sphere(radius=0.01)
        add_soft_obstacle(plant, location, mesh, name)

    if scenario == "wall" or scenario == "both":
        name = "wall"
        location = RigidTransform(np.array([0.3, 0.33, 0]))
        if randomise_obstacle:
            location = RigidTransform(
                np.array(
                    [
                        rng.uniform(0.1, 0.5),
                        rng.uniform(0.31, 0.35),
                        0,
                    ]
                )
            )
        mesh = Box(0.15, 0.05, 3.0)
        add_soft_arm_collisions(plant)
        add_soft_obstacle(plant, location, mesh, name)

    if scenario == "None":
        pass
    # -----------------------------------------
    plant.Finalize()
    nq = plant.num_positions()  # 30
    nv = plant.num_velocities()  # 29
    na = plant.num_actuators()  # 23
    # -----------------------------------------
    # for visualisation only
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        ContactVisualizer.AddToBuilder(
            builder,
            plant,
            meshcat,
            ContactVisualizerParams(newtons_per_meter=1e3, newton_meters_per_meter=1e1),
        )
    # -----------------------------------------
    tsid_params = TSIDParameterTuning()
    tsid_params.set_from_x_k(x_k)
    tsid_controller = builder.AddSystem(
        QPfloatingTSID(plant, robot_model_con, tsid_params, to_lock, None, qRemap)
    )
    # -----------------------------------------
    mpc_parameters = MPCParameterTuning()
    mpc_controller = builder.AddSystem(
        CentroidalMPC(plant, robot_model_con, mpc_parameters, qRemap)
    )
    demux1 = builder.AddSystem(
        Demultiplexer([6, 3, 3, 3])
    )  # (com, dcom) + left-forces + right-forces + ang_mom
    demux2 = builder.AddSystem(Demultiplexer([20, 20]))  # lfoot and rfoot references
    # -----------------------------------------
    if add_base_noise != 0.0:
        noise_adder = builder.AddSystem(
            NoiseAdder(
                plant,
                size=nq + nv,
                idx=np.array([4, 5, 6]),
                rng=rng,
                noise_type="max",
                noise_params={
                    "min": np.array([0.0] * 3),
                    "max": np.array([add_base_noise] * 3),
                },
            )
        )
        builder.Connect(
            plant.get_state_output_port(robot_model_sim), noise_adder.get_input_port()
        )
        builder.Connect(
            noise_adder.get_output_port(), tsid_controller.GetInputPort("robot_state")
        )
        builder.Connect(
            noise_adder.get_output_port(), mpc_controller.GetInputPort("robot_state")
        )
    else:
        builder.Connect(
            plant.get_state_output_port(robot_model_sim),
            tsid_controller.GetInputPort("robot_state"),
        )
        builder.Connect(
            plant.get_state_output_port(robot_model_sim),
            mpc_controller.GetInputPort("robot_state"),
        )

    builder.Connect(mpc_controller.get_output_port(0), demux1.get_input_port())
    builder.Connect(mpc_controller.get_output_port(1), demux2.get_input_port())

    builder.Connect(
        demux1.get_output_port(0), tsid_controller.GetInputPort("com_desired")
    )
    builder.Connect(
        demux1.get_output_port(1), tsid_controller.GetInputPort("left_wrench_desired")
    )
    builder.Connect(
        demux1.get_output_port(2), tsid_controller.GetInputPort("right_wrench_desired")
    )
    builder.Connect(
        demux1.get_output_port(3),
        tsid_controller.GetInputPort("centroidal_ang_mom_desired"),
    )
    builder.Connect(
        demux2.get_output_port(0),
        tsid_controller.GetInputPort("left_foot_pose_desired"),
    )
    builder.Connect(
        demux2.get_output_port(1),
        tsid_controller.GetInputPort("right_foot_pose_desired"),
    )
    # -----------------------------------------
    if add_root_disturbance != 0.0:
        disturbance_sys = builder.AddSystem(
            DisturbanceSystem(
                plant,
                "root_link",
                np.concatenate(
                    (
                        np.zeros(3),  # moments
                        rng.uniform(-add_root_disturbance, add_root_disturbance, 3),
                    )
                ),
                5,  # apply every 5 s
                0.01,  # for 0.01 s
            )
        )
        builder.Connect(
            disturbance_sys.get_output_port(),
            plant.get_applied_spatial_force_input_port(),
        )
    # -----------------------------------------
    joint_actuation = JointActuation(
        plant,
        gear_ratio_matrix,  # gear ratios
        motor_inertia_matrix,  # motor inertia
        tau_sat,  # torque saturation
        f_K0,  # f_K0
        f_K1,  # f_K1
        f_K2,  # f_K2
        np.zeros((na, na)),  # K_bemf
        delay=1e-3,  # delay -- 1 ms
    )
    builder.AddSystem(joint_actuation)
    builder.Connect(
        plant.get_state_output_port(robot_model_sim),
        joint_actuation.GetInputPort("robot_state"),
    )
    builder.Connect(
        tsid_controller.get_output_port(),
        joint_actuation.GetInputPort("demanded_torque"),
    )
    zoh = builder.AddSystem(ZeroOrderHold(0.0, nv))
    builder.Connect(
        plant.get_generalized_acceleration_output_port(), zoh.get_input_port()
    )
    builder.Connect(
        zoh.get_output_port(), joint_actuation.GetInputPort("generalised_accelerations")
    )
    builder.Connect(
        joint_actuation.GetOutputPort("realised_torque"),
        plant.get_actuation_input_port(),
    )
    # -----------------------------------------
    if meshcat is not None:
        taudlog = psp.LogVectorOutput(tsid_controller.get_output_port(), builder)
        taurlog = psp.LogVectorOutput(
            joint_actuation.GetOutputPort("realised_torque"), builder
        )
        qlog = psp.LogVectorOutput(
            plant.get_state_output_port(robot_model_sim), builder
        )
    tau_flog = psp.LogVectorOutput(
        joint_actuation.GetOutputPort("friction_torque"), builder
    )
    tau_mlog = psp.LogVectorOutput(
        joint_actuation.GetOutputPort("motor_torque"), builder
    )
    mv_log = psp.LogVectorOutput(
        joint_actuation.GetOutputPort("motor_velocity"), builder
    )
    # -----------------------------------------
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    [plant.GetJointByName(jj).Lock(plant_context) for jj in to_lock]
    tsid_context = tsid_controller.GetMyMutableContextFromRoot(diagram_context)
    mpc_context = mpc_controller.GetMyMutableContextFromRoot(diagram_context)
    plant.SetPositions(
        plant_context, np.concatenate((base_ori_init_wxyz, base_pos_init, s_init_sim))
    )
    mpc_controller.GetInputPort("com0").FixValue(mpc_context, com_init)
    tsid_controller.GetInputPort("joint_angle_desired").FixValue(
        tsid_context, s_init_sim
    )
    tsid_controller.GetInputPort("root_ori_desired").FixValue(
        tsid_context, root_link_ori_wxyz
    )
    if meshcat is not None:
        meshcat.Delete()
        meshcat.DeleteAddedControls()
    simulator = Simulator(diagram, diagram_context)

    def monitor(context):  # monitor MPC and TSID flags
        if tsid_controller.solve_status and mpc_controller.solve_status:
            return EventStatus.Succeeded()
        else:
            return EventStatus.ReachedTermination(diagram, "Controller failed")

    simulator.set_monitor(monitor)
    Tf = 20.0

    # plt.figure(figsize=(11, 8.5), dpi=300)
    # plt.figure(dpi=600)
    # plot_system_graphviz(diagram)
    # plt.savefig("run_walking_sim_diagram.png")

    try:
        if meshcat is not None:
            meshcat.StartRecording()
        simulator.AdvanceTo(Tf)
        if meshcat is not None:
            meshcat.StopRecording()
    except RuntimeError as e:
        logging.error("Simulation failed with RuntimeError:\n {}".format(e))
    sim_context = simulator.get_context()
    Tf = sim_context.get_time()
    # -----------------------------------------
    if meshcat is not None:
        meshcat.PublishRecording()
        try:
            plot_data(qlog.FindLog(diagram_context), Tf, "q")
            plot_data(qlog.FindLog(diagram_context), Tf, "v")
            plot_data(taudlog.FindLog(diagram_context), Tf, "u")
            plot_data(taurlog.FindLog(diagram_context), Tf, "u")
        except:
            pass

    return (
        Tf,
        tau_mlog.FindLog(diagram_context),
        tau_flog.FindLog(diagram_context),
        mv_log.FindLog(diagram_context),
        motor_constant_matrix,
    )
