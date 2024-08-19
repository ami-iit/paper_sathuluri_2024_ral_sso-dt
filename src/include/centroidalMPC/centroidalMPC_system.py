import logging
from copy import copy as pycopy
from datetime import timedelta

import bipedal_locomotion_framework as blf
import idyntree.bindings as idyn
import matplotlib.pyplot as plt
import numpy as np
from include.centroidalMPC.footPositionPlanner import FootPositionPlanner
from pydrake.common.eigen_geometry import Quaternion
from pydrake.math import RigidTransform
from pydrake.systems.framework import BasicVector, LeafSystem

logging.basicConfig(level=logging.CRITICAL)


class CentroidalMPC(LeafSystem):
    def __init__(self, plant, robot_model, mpc_parameters, qRemap):
        LeafSystem.__init__(self)
        self.state_var_idx1 = self.DeclareDiscreteState(15)
        self.state_var_idx2 = self.DeclareDiscreteState(40)

        self.dT_planner = 5e-3
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=self.dT_planner,
            offset_sec=self.dT_planner,
            update=self.update_foot_states,
        )
        logging.info("Updating footstep planner at: {} Hz".format(1 / self.dT_planner))

        self.dT = 0.1  # every 100 ms
        self.DeclareInitializationDiscreteUpdateEvent(update=self.initialise_controller)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=self.dT, offset_sec=0.0, update=self.update_system_states
        )
        logging.info("Updating centroidalMPC at: {} Hz".format(1 / self.dT))

        self.contact_planner = FootPositionPlanner(
            robot_model=robot_model, dT=self.dT_planner
        )
        self.centroidal_mpc = blf.reduced_model_controllers.CentroidalMPC()

        scaling = 1.0
        scalingPos = 1.2
        scalingPosY = 0.0

        self.gravity = idyn.Vector3()
        self.gravity.zero()
        self.gravity.setVal(2, -blf.math.StandardAccelerationOfGravitation)
        self.contact_planner.set_scaling_parameters(
            scaling=scaling, scalingPos=scalingPos, scalingPosY=scalingPosY
        )
        self.kindyn = robot_model.get_idyntree_kyndyn()
        self.total_mass = robot_model.get_total_mass()

        self.robot_model = robot_model
        self.plant = plant
        self.qRemap = qRemap
        self.nq = plant.num_positions()
        self.nv = plant.num_velocities()
        self.na = plant.num_actuators()
        self.mpc_parameters = mpc_parameters
        self.solve_status = True

        self.DeclareVectorInputPort(
            "robot_state", BasicVector(plant.num_positions() + plant.num_velocities())
        )
        self.DeclareVectorInputPort("com0", BasicVector(3))
        self.DeclareVectorOutputPort(
            "mpc-references", BasicVector(15), self.SetMPCOutputReference
        )
        self.DeclareVectorOutputPort(
            "foot-references", BasicVector(40), self.SetPlannerOutputReference
        )
        self.DeclareVectorOutputPort("com-obs", BasicVector(6), self.CoMObserved)

    def CoMObserved(self, context, output):
        output.SetFromVector(np.concatenate((self.com, self.dcom)))

    def initialize_mpc(self, mpc_parameters):
        time_horizon = self.mpc_parameters.time_horizon
        self.mpc_param_handler = blf.parameters_handler.StdParametersHandler()
        self.mpc_param_handler.set_parameter_datetime(
            "sampling_time", timedelta(seconds=self.dT)
        )
        self.mpc_param_handler.set_parameter_datetime("time_horizon", time_horizon)
        self.mpc_param_handler.set_parameter_int("number_of_maximum_contacts", 2)
        self.mpc_param_handler.set_parameter_vector_float(
            "com_weight", mpc_parameters.com_weight
        )
        self.mpc_param_handler.set_parameter_float(
            "contact_position_weight", mpc_parameters.contact_position_weight
        )
        self.mpc_param_handler.set_parameter_vector_float(
            "force_rate_of_change_weight", mpc_parameters.force_rate_change_weight
        )
        self.mpc_param_handler.set_parameter_float(
            "angular_momentum_weight", mpc_parameters.angular_momentum_weight
        )
        self.mpc_param_handler.set_parameter_float(
            "contact_force_symmetry_weight",
            mpc_parameters.contact_force_symmetry_weight,
        )
        self.mpc_param_handler.set_parameter_int("number_of_slices", 1)
        self.mpc_param_handler.set_parameter_float("static_friction_coefficient", 0.33)
        self.mpc_param_handler.set_parameter_string("linear_solver", "ma97")
        self.mpc_param_handler.set_parameter_int("verbosity", 0)

        self.contact_0_handler = blf.parameters_handler.StdParametersHandler()
        self.contact_0_handler.set_parameter_string("contact_name", "left_foot")
        self.contact_0_handler.set_parameter_vector_float(
            "bounding_box_lower_limit", [0.0, 0.0, 0.0]
        )
        self.contact_0_handler.set_parameter_vector_float(
            "bounding_box_upper_limit", [0.0, 0.0, 0.0]
        )
        self.contact_0_handler.set_parameter_int("number_of_corners", 4)
        self.contact_0_handler.set_parameter_vector_float("corner_0", [0.08, 0.03, 0.0])
        self.contact_0_handler.set_parameter_vector_float(
            "corner_1", [0.08, -0.03, 0.0]
        )
        self.contact_0_handler.set_parameter_vector_float(
            "corner_2", [-0.08, -0.03, 0.0]
        )
        self.contact_0_handler.set_parameter_vector_float(
            "corner_3", [-0.08, 0.03, 0.0]
        )

        self.contact_1_handler = blf.parameters_handler.StdParametersHandler()
        self.contact_1_handler.set_parameter_string("contact_name", "right_foot")
        self.contact_1_handler.set_parameter_vector_float(
            "bounding_box_lower_limit", [0.0, 0.0, 0.0]
        )
        self.contact_1_handler.set_parameter_vector_float(
            "bounding_box_upper_limit", [0.0, 0.0, 0.0]
        )
        self.contact_1_handler.set_parameter_int("number_of_corners", 4)
        self.contact_1_handler.set_parameter_vector_float("corner_0", [0.08, 0.03, 0.0])
        self.contact_1_handler.set_parameter_vector_float(
            "corner_1", [0.08, -0.03, 0.0]
        )
        self.contact_1_handler.set_parameter_vector_float(
            "corner_2", [-0.08, -0.03, 0.0]
        )
        self.contact_1_handler.set_parameter_vector_float(
            "corner_3", [-0.08, 0.03, 0.0]
        )

        self.mpc_param_handler.set_group("CONTACT_0", self.contact_0_handler)
        self.mpc_param_handler.set_group("CONTACT_1", self.contact_1_handler)
        self.centroidal_mpc.initialize(self.mpc_param_handler)

    def configure(self):
        self.contact_planner.update_initial_position(self.H_b, self.s)
        self.contact_planner.compute_feet_contact_position()
        self.contact_phase_list = self.contact_planner.get_contact_phase_list()
        self.centroidal_mpc.set_contact_phase_list(self.contact_phase_list)
        self.contact_planner.initialize_foot_swing_planner()

    def define_test_com_traj(self, context):
        com_knots = []
        time_knots = []
        com0 = self.kindyn.getCenterOfMassPosition().toNumPy()
        self.com0 = pycopy(com0)
        com_knots.append(com0)
        vector_phase_list = self.contact_phase_list
        time_knots.append(vector_phase_list.first_phase().begin_time)

        for item in vector_phase_list:
            if (
                len(item.active_contacts) == 2
                and not (vector_phase_list.first_phase() is item)
                and not (vector_phase_list.last_phase() is item)
            ):
                time_knots.append((item.end_time + item.begin_time) / 2)
                p1 = item.active_contacts["left_foot"].pose.translation()
                p2 = item.active_contacts["right_foot"].pose.translation()
                des_com = (p1 + p2) / 2
                des_com[2] = com0[2]
                com_knots.append(des_com)
            elif len(item.active_contacts) == 2 and (
                vector_phase_list.last_phase() is item
            ):
                time_knots.append(item.end_time)
                p1 = item.active_contacts["left_foot"].pose.translation()
                p2 = item.active_contacts["right_foot"].pose.translation()
                des_com = (p1 + p2) / 2
                des_com[2] = com0[2]
                com_knots.append(des_com)

        com_spline = blf.planners.QuinticSpline()
        com_spline.set_initial_conditions(np.zeros(3), np.zeros(3))
        com_spline.set_final_conditions(np.zeros(3), np.zeros(3))
        com_spline.set_knots(com_knots, time_knots)
        tempInt = 1000

        com_traj = []
        angular_mom_traj = []
        velocity = np.zeros(3)
        acceleration = np.zeros(3)
        for i in range(tempInt):
            angular_mom_traj_i = np.zeros(3)
            com_temp = np.zeros(3)
            com_spline.evaluate_point(i * self.dT, com_temp, velocity, acceleration)
            com_traj.append(com_temp)
            angular_mom_traj.append(angular_mom_traj_i)

        self.centroidal_mpc.set_reference_trajectory(com_traj, angular_mom_traj)
        self.centroidal_mpc.set_contact_phase_list(vector_phase_list)
        self.com_traj = com_traj
        self.angular_mom_trak = angular_mom_traj

    def update_contact_phase_list(self, next_planned_contacts):
        new_contact_list = self.contact_phase_list.lists()

        for key, contact in next_planned_contacts:
            it = new_contact_list[key].get_present_contact(contact.activation_time)
            new_contact_list[key].edit_contact(it, contact)

        self.contact_phase_list.set_lists(new_contact_list)

    def initialize_centroidal_integrator(self):
        self.centroidal_integrator = (
            blf.continuous_dynamical_system.CentroidalDynamicsForwardEulerIntegrator()
        )
        self.centroidal_dynamics = blf.continuous_dynamical_system.CentroidalDynamics()
        self.centroidal_integrator.set_dynamical_system(self.centroidal_dynamics)

        com = self.kindyn.getCenterOfMassPosition().toNumPy()
        dcom = self.kindyn.getCenterOfMassVelocity().toNumPy()

        total_mom = self.kindyn.getCentroidalTotalMomentum().toNumPy()
        angular_mom = total_mom[3:] / self.total_mass
        self.centroidal_dynamics.set_state((com, dcom, angular_mom))
        # ---------------
        self.centroidal_integrator.set_integration_step(timedelta(seconds=self.dT))

    def plan_trajectory(self, context):
        com = self.kindyn.getCenterOfMassPosition().toNumPy()

        dcom = self.kindyn.getCenterOfMassVelocity().toNumPy()
        total_mom = self.kindyn.getCentroidalTotalMomentum().toNumPy()
        angular_mom = total_mom[3:] / self.total_mass

        self.com = com
        self.dcom = dcom

        self.centroidal_mpc.set_state(com, dcom, angular_mom)
        self.centroidal_mpc.set_reference_trajectory(
            self.com_traj, self.angular_mom_trak
        )
        self.centroidal_mpc.set_contact_phase_list(self.contact_phase_list)
        success = self.centroidal_mpc.advance()
        if success:
            self.centroidal_dynamics.set_control_input(
                (self.centroidal_mpc.get_output().contacts, np.zeros(6))
            )
            self.centroidal_integrator.integrate(
                timedelta(0), timedelta(seconds=self.dT)
            )

        return success

    def update_com_reference(self):
        self.com_traj = self.com_traj[1:]

    def update_robot_state(self, context):
        rs = self.GetInputPort("robot_state").Eval(context)
        q = rs[: self.nq]
        base_quat_wxyz = q[:4]
        base_quat_wxyz_norm = Quaternion(
            base_quat_wxyz / (np.linalg.norm(base_quat_wxyz))
        )
        base_pos = q[4:7]
        s_sim = q[7:]
        dq = rs[self.nq :]
        base_ang_vel = dq[:3]
        base_lin_vel = dq[3:6]
        ds_sim = dq[6:]

        self.H_b = RigidTransform(base_quat_wxyz_norm, base_pos).GetAsMatrix4()
        self.s = self.qRemap(s_sim, "sim")
        self.w_b = np.concatenate((base_lin_vel, base_ang_vel))
        self.s_dot = self.qRemap(ds_sim, "sim")
        self.kindyn.setRobotState(self.H_b, self.s, self.w_b, self.s_dot, self.gravity)

    def initialise_controller(self, context, discrete_state):
        self.update_robot_state(context)
        logging.info("Initialising MPC")
        self.initialize_mpc(self.mpc_parameters)
        self.configure()
        self.define_test_com_traj(context)
        self.initialize_centroidal_integrator()
        mpc_success = self.plan_trajectory(context)
        logging.info("MPC initialisation status: {}".format(mpc_success))
        self.contact_planner.advance_swing_foot_planner()
        (
            self.ref_left_foot,
            self.ref_right_foot,
        ) = self.contact_planner.get_references_swing_foot_planner()
        lfoot_ref_vec = np.concatenate(
            (
                Quaternion(self.ref_left_foot.transform.rotation()).wxyz(),
                self.ref_left_foot.transform.translation(),
                self.ref_left_foot.mixed_velocity,
                self.ref_left_foot.mixed_acceleration,
                np.array([int(self.ref_left_foot.is_in_contact)]),
            )
        )
        rfoot_ref_vec = np.concatenate(
            (
                Quaternion(self.ref_right_foot.transform.rotation()).wxyz(),
                self.ref_right_foot.transform.translation(),
                self.ref_right_foot.mixed_velocity,
                self.ref_right_foot.mixed_acceleration,
                np.array([int(self.ref_right_foot.is_in_contact)]),
            )
        )
        lf_ref_vec = np.concatenate((lfoot_ref_vec, rfoot_ref_vec))
        discrete_state.get_mutable_vector(self.state_var_idx2).set_value(lf_ref_vec)
        if mpc_success:
            self._is_initialised = True
            com, dcom, forces_left, forces_right, ang_mom = self.get_references()
            all_refs = np.concatenate((com, dcom, forces_left, forces_right, ang_mom))
            discrete_state.get_mutable_vector(self.state_var_idx1).set_value(all_refs)
        else:
            logging.error("MPC system failed to initialise")

    def update_system_states(self, context, discrete_state):
        self.update_robot_state(context)
        self.update_com_reference()
        mpc_success = self.plan_trajectory(context)
        if not mpc_success:
            logging.error("MPC stepping failed")
        self.solve_status = mpc_success

        com, dcom, forces_left, forces_right, ang_mom = self.get_references()
        all_refs = np.concatenate((com, dcom, forces_left, forces_right, ang_mom))
        discrete_state.get_mutable_vector(self.state_var_idx1).set_value(all_refs)

    def SetMPCOutputReference(self, context, output):
        output.SetFromVector(context.get_discrete_state(self.state_var_idx1).value())

    def update_foot_states(self, context, discrete_state):
        self.contact_planner.advance_swing_foot_planner()
        (
            self.ref_left_foot,
            self.ref_right_foot,
        ) = self.contact_planner.get_references_swing_foot_planner()
        lfoot_ref_vec = np.concatenate(
            (
                Quaternion(self.ref_left_foot.transform.rotation()).wxyz(),
                self.ref_left_foot.transform.translation(),
                self.ref_left_foot.mixed_velocity,
                self.ref_left_foot.mixed_acceleration,
                np.array([int(self.ref_left_foot.is_in_contact)]),
            )
        )
        rfoot_ref_vec = np.concatenate(
            (
                Quaternion(self.ref_right_foot.transform.rotation()).wxyz(),
                self.ref_right_foot.transform.translation(),
                self.ref_right_foot.mixed_velocity,
                self.ref_right_foot.mixed_acceleration,
                np.array([int(self.ref_right_foot.is_in_contact)]),
            )
        )
        lf_ref_vec = np.concatenate((lfoot_ref_vec, rfoot_ref_vec))
        discrete_state.get_mutable_vector(self.state_var_idx2).set_value(lf_ref_vec)

    def SetPlannerOutputReference(self, context, output):
        output.SetFromVector(context.get_discrete_state(self.state_var_idx2).value())

    def get_references(self):
        output_mpc = self.get_output()
        left_foot_tag = "left_foot"
        right_foot_tag = "right_foot"
        forces_left = np.zeros(3)
        for item in output_mpc.contacts[left_foot_tag].corners:
            forces_left = forces_left + item.force
        forces_right = np.zeros(3)
        for item in output_mpc.contacts[right_foot_tag].corners:
            forces_right = forces_right + item.force

        com = self.centroidal_integrator.get_solution()[0]
        dcom = self.centroidal_integrator.get_solution()[1]
        ang_mom = self.centroidal_integrator.get_solution()[2]

        # -------------
        forces_left *= self.total_mass
        forces_right *= self.total_mass
        ang_mom *= self.total_mass
        # -------------
        return com, dcom, forces_left, forces_right, ang_mom

    def get_output(self):
        return self.centroidal_mpc.get_output()

    def plot_3d_foot(self):
        left_foot_tag = "left_foot"
        right_foot_tag = "right_foot"
        fig = plt.figure()
        self.left_foot_fig = fig.add_subplot(111, projection="3d")
        fig = plt.figure()
        self.right_foot_fig = fig.add_subplot(111, projection="3d")
        output_mpc = self.get_output()
        contact_left = output_mpc.contacts[left_foot_tag]
        contact_right = output_mpc.contacts[right_foot_tag]
        for item in contact_left.corners:
            self.left_foot_fig.quiver(
                item.position[0],
                item.position[1],
                item.position[2],
                item.force[0],
                item.force[1],
                item.force[2],
            )
        for item in contact_right.corners:
            self.right_foot_fig.quiver(
                item.position[0],
                item.position[1],
                item.position[2],
                item.force[0],
                item.force[1],
                item.force[2],
            )
        self.right_foot_fig.set_xlim([-30, 30])
        self.right_foot_fig.set_ylim([-30, 30])
        self.right_foot_fig.set_zlim([-50, 50])
        self.left_foot_fig.set_xlim([-30, 30])
        self.left_foot_fig.set_ylim([-30, 30])
        self.left_foot_fig.set_zlim([-50, 50])
        plt.show()

    def plot_com_no_meaning(self, dT):
        T = 0
        com_list = []
        while T < 30:
            self.update_references()
            mpc_success = self.plan_trajectory()
            self.contact_planner.advance_swing_foot_planner()
            com, dcom, forces_left, forces_right = self.get_references()
            com_list.append(com)
            T += dT
        com_list = np.array(com_list)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(com_list[:, 0], com_list[:, 1], com_list[:, 2])
        plt.title("CoM trajectory vs time predicted by Centroidal MPC")
        fig.text(
            0.5,
            0.05,
            "The image is purely to test the MPC predictor and has no meaning as the state is not updated",
            ha="center",
        )
        plt.show()
