import bipedal_locomotion_framework as blf
import idyntree.bindings as idyn
import manifpy as manif

# %%
import numpy as np
from IPython.display import clear_output
from pydrake.common.eigen_geometry import Quaternion
from pydrake.math import RigidTransform
from pydrake.systems.framework import BasicVector, LeafSystem
import logging

logging.basicConfig(level=logging.CRITICAL)


# %%
class QPfloatingTSID(LeafSystem):
    def __init__(self, plant, robot_model, tsid_params, to_lock, viz, qRemap):
        blf.text_logging.set_verbosity(blf.text_logging.Verbosity.Off)
        LeafSystem.__init__(self)
        self.controller = blf.tsid.QPTSID()
        self.qRemap = qRemap
        self.gravity = idyn.Vector3()
        self.gravity.zero()
        self.gravity.setVal(2, -blf.math.StandardAccelerationOfGravitation)
        self.kindyn = robot_model.get_idyntree_kyndyn()
        self.robot_acceleration_variable_name = "robotAcceleration"
        self.joint_torques_variable_name = "joint_torques"
        self.variable_name_right_contact = "rf_wrench"
        self.variable_name_left_contact = "lf_wrench"
        self.torque = None
        self.plant = plant
        self.robot_model = robot_model
        self.nq = plant.num_positions()
        self.nv = plant.num_velocities()
        self.na = plant.num_actuators()
        self.na_con = self.na - len(to_lock)
        self.max_number_contacts = 2
        self.tau_state = self.DeclareDiscreteState(plant.num_actuators())
        self.dT = 0.01
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=self.dT, offset_sec=0.0, update=self.UpdateOutputJointTorque
        )
        logging.info("Updating TSID at: {} Hz".format(1 / self.dT))
        self.solve_status = True

        self.DeclareVectorInputPort(
            "robot_state", BasicVector(plant.num_positions() + plant.num_velocities())
        )
        self.DeclareVectorInputPort(
            "joint_angle_desired", BasicVector(plant.num_actuators())
        )
        self.DeclareVectorInputPort("com_desired", BasicVector(6))
        self.DeclareVectorInputPort("left_wrench_desired", BasicVector(3))
        self.DeclareVectorInputPort("right_wrench_desired", BasicVector(3))
        self.DeclareVectorInputPort("left_foot_pose_desired", BasicVector(20))
        self.DeclareVectorInputPort("right_foot_pose_desired", BasicVector(20))
        self.DeclareVectorInputPort("centroidal_ang_mom_desired", BasicVector(3))
        self.DeclareVectorInputPort("root_ori_desired", BasicVector(4))

        self.DeclareVectorOutputPort(
            "tau", BasicVector(plant.num_actuators()), self.OutputJointTorque
        )
        self.define_tasks(tsid_params)

    def define_varible_handler(self):
        self.var_handler = blf.system.VariablesHandler()
        self.var_handler.add_variable(
            self.robot_acceleration_variable_name, self.robot_model.NDoF + 6
        )
        self.var_handler.add_variable(
            self.joint_torques_variable_name, self.robot_model.NDoF
        )
        self.var_handler.add_variable(self.variable_name_right_contact, 6)
        self.var_handler.add_variable(self.variable_name_left_contact, 6)

    def define_tasks(self, tsid_parameters):
        self.controller_variable_handler = blf.parameters_handler.StdParametersHandler()
        self.controller_variable_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.controller_variable_handler.set_parameter_string(
            name="joint_torques_variable_name", value=self.joint_torques_variable_name
        )
        contact_name_list = [
            self.variable_name_left_contact,
            self.variable_name_right_contact,
        ]
        self.controller_variable_handler.set_parameter_vector_string(
            name="contact_wrench_variables_name", value=contact_name_list
        )
        self.controller.initialize(self.controller_variable_handler)
        self.define_varible_handler()

        # create contact groups
        contact_group_left = blf.parameters_handler.StdParametersHandler()
        contact_group_left.set_parameter_string(
            name="variable_name", value=self.variable_name_left_contact
        )
        contact_group_left.set_parameter_string(
            name="frame_name", value=self.robot_model.left_foot_frame
        )
        contact_group_right = blf.parameters_handler.StdParametersHandler()
        contact_group_right.set_parameter_string(
            name="variable_name", value=self.variable_name_right_contact
        )
        contact_group_right.set_parameter_string(
            name="frame_name", value=self.robot_model.right_foot_frame
        )

        self.joint_tracking_task = blf.tsid.JointTrackingTask()
        self.joint_tracking_param_handler = (
            blf.parameters_handler.StdParametersHandler()
        )
        self.joint_tracking_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.joint_tracking_param_handler.set_parameter_vector_float(
            name="kp", value=tsid_parameters.postural_Kp
        )
        self.joint_tracking_param_handler.set_parameter_vector_float(
            name="kd", value=tsid_parameters.postural_Kd
        )
        self.joint_tracking_task_name = "joint_tracking_task"
        self.joint_tracking_task_priority = 1
        self.joint_tracking_task.set_kin_dyn(self.kindyn)
        self.joint_tracking_task.initialize(self.joint_tracking_param_handler)
        self.joint_tracking_task_weight = tsid_parameters.postural_weight

        self.joint_dynamics_task = blf.tsid.JointDynamicsTask()
        self.joint_dynamics_task_name = "joint_dynamics_task"
        self.joint_dynamics_task_priority = 0
        self.joint_dynamics_task_weight = 1e2 * np.ones(self.na_con)
        self.joint_dynamics_task_param_handler = (
            blf.parameters_handler.StdParametersHandler()
        )
        self.joint_dynamics_task_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.joint_dynamics_task_param_handler.set_parameter_string(
            name="joint_torques_variable_name", value=self.joint_torques_variable_name
        )
        self.joint_dynamics_task_param_handler.set_parameter_int(
            name="max_number_of_contacts", value=2
        )
        self.joint_dynamics_task_param_handler.set_group(
            "CONTACT_0", contact_group_left
        )
        self.joint_dynamics_task_param_handler.set_group(
            "CONTACT_1", contact_group_right
        )
        self.joint_dynamics_task.set_kin_dyn(self.kindyn)
        self.joint_dynamics_task.initialize(
            param_handler=self.joint_dynamics_task_param_handler
        )

        self.CoM_Task = blf.tsid.CoMTask()
        self.CoM_task_name = "CoM_task"
        self.CoM_task_priority = 0
        self.CoM_task_weight = 1e5 * np.ones(3)
        self.CoM_param_handler = blf.parameters_handler.StdParametersHandler()
        self.CoM_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.CoM_param_handler.set_parameter_float(
            name="kp_linear", value=tsid_parameters.CoM_Kp
        )
        self.CoM_param_handler.set_parameter_float(
            name="kd_linear", value=tsid_parameters.CoM_Kd
        )
        self.CoM_Task.set_kin_dyn(self.kindyn)
        self.CoM_Task.initialize(param_handler=self.CoM_param_handler)

        self.base_dynamic_task = blf.tsid.BaseDynamicsTask()
        self.base_dynamic_task_name = "base_dynamic_task"
        self.base_dynamic_task_priority = 0
        self.base_dynamic_param_handler = blf.parameters_handler.StdParametersHandler()
        self.base_dynamic_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.base_dynamic_param_handler.set_parameter_int(
            name="max_number_of_contacts", value=self.max_number_contacts
        )
        self.base_dynamic_param_handler.set_group("CONTACT_0", contact_group_left)
        self.base_dynamic_param_handler.set_group("CONTACT_1", contact_group_right)
        self.base_dynamic_task.set_kin_dyn(self.kindyn)
        self.base_dynamic_task.initialize(param_handler=self.base_dynamic_param_handler)
        self.base_dynamic_task_weight = np.ones(6) * 1e1

        self.left_foot_tracking_task = blf.tsid.SE3Task()
        self.left_foot_tracking_task_name = "left_foot_tracking_task"
        self.left_foot_tracking_task_priority = 0
        self.left_foot_tracking_task_param_handler = (
            blf.parameters_handler.StdParametersHandler()
        )
        self.left_foot_tracking_task_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.left_foot_tracking_task_param_handler.set_parameter_string(
            name="frame_name", value=self.robot_model.left_foot_frame
        )
        self.left_foot_tracking_task_param_handler.set_parameter_float(
            name="kp_linear", value=tsid_parameters.foot_tracking_task_kp_lin
        )
        self.left_foot_tracking_task_param_handler.set_parameter_float(
            name="kd_linear", value=tsid_parameters.foot_tracking_task_kd_lin
        )
        self.left_foot_tracking_task_param_handler.set_parameter_float(
            name="kp_angular", value=tsid_parameters.foot_tracking_task_kp_ang
        )
        self.left_foot_tracking_task_param_handler.set_parameter_float(
            name="kd_angular", value=tsid_parameters.foot_tracking_task_kd_ang
        )
        self.left_foot_tracking_task.set_kin_dyn(self.kindyn)
        self.left_foot_tracking_task.initialize(
            param_handler=self.left_foot_tracking_task_param_handler
        )

        self.right_foot_tracking_task = blf.tsid.SE3Task()
        self.right_foot_tracking_task_name = "right_foot_tracking_task"
        self.right_foot_tracking_task_priority = 0
        self.right_foot_tracking_task_param_handler = (
            blf.parameters_handler.StdParametersHandler()
        )
        self.right_foot_tracking_task_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.right_foot_tracking_task_param_handler.set_parameter_string(
            name="frame_name", value=self.robot_model.right_foot_frame
        )
        self.right_foot_tracking_task_param_handler.set_parameter_float(
            name="kp_linear", value=tsid_parameters.foot_tracking_task_kp_lin
        )
        self.right_foot_tracking_task_param_handler.set_parameter_float(
            name="kd_linear", value=tsid_parameters.foot_tracking_task_kd_lin
        )
        self.right_foot_tracking_task_param_handler.set_parameter_float(
            name="kp_angular", value=tsid_parameters.foot_tracking_task_kp_ang
        )
        self.right_foot_tracking_task_param_handler.set_parameter_float(
            name="kd_angular", value=tsid_parameters.foot_tracking_task_kd_ang
        )
        self.right_foot_tracking_task.set_kin_dyn(self.kindyn)
        self.right_foot_tracking_task.initialize(
            param_handler=self.right_foot_tracking_task_param_handler
        )

        self.left_foot_wrench_task = blf.tsid.FeasibleContactWrenchTask()
        self.left_foot_wrench_task_name = "left_foot_wrench_task"
        self.left_foot_wrench_priority = 0
        self.left_foot_wrench_task_weight = np.ones(3)
        self.left_foot_param_handler = blf.parameters_handler.StdParametersHandler()
        self.left_foot_param_handler.set_parameter_string(
            name="variable_name", value=self.variable_name_left_contact
        )
        self.left_foot_param_handler.set_parameter_string(
            name="frame_name", value=self.robot_model.left_foot_frame
        )
        self.left_foot_param_handler.set_parameter_int(name="number_of_slices", value=2)
        self.left_foot_param_handler.set_parameter_float(
            name="static_friction_coefficient", value=1.0
        )
        self.left_foot_param_handler.set_parameter_vector_float(
            name="foot_limits_x", value=[-0.08, 0.08]
        )
        self.left_foot_param_handler.set_parameter_vector_float(
            name="foot_limits_y", value=[-0.03, 0.03]
        )
        self.left_foot_wrench_task.set_kin_dyn(self.kindyn)
        self.left_foot_wrench_task.initialize(
            param_handler=self.left_foot_param_handler
        )

        self.right_foot_wrench_task = blf.tsid.FeasibleContactWrenchTask()
        self.right_foot_wrench_task_name = "right_foot_wrench_task"
        self.right_foot_wrench_priority = 0
        self.right_foot_wrench_task_weight = np.ones(3)
        self.right_foot_param_handler = blf.parameters_handler.StdParametersHandler()
        self.right_foot_param_handler.set_parameter_string(
            name="variable_name", value=self.variable_name_right_contact
        )
        self.right_foot_param_handler.set_parameter_string(
            name="frame_name", value=self.robot_model.right_foot_frame
        )
        self.right_foot_param_handler.set_parameter_int(
            name="number_of_slices", value=2
        )
        self.right_foot_param_handler.set_parameter_float(
            name="static_friction_coefficient", value=1.0
        )
        self.right_foot_param_handler.set_parameter_vector_float(
            name="foot_limits_x", value=[-0.08, 0.08]
        )
        self.right_foot_param_handler.set_parameter_vector_float(
            name="foot_limits_y", value=[-0.03, 0.03]
        )
        self.right_foot_wrench_task.set_kin_dyn(self.kindyn)
        self.right_foot_wrench_task.initialize(
            param_handler=self.right_foot_param_handler
        )

        self.root_link_task = blf.tsid.SO3Task()
        self.root_link_task_name = "root_link_task"
        self.root_link_task_priority = 1
        self.root_link_task_weight = tsid_parameters.root_tracking_task_weight
        self.root_link_task_param_handler = (
            blf.parameters_handler.StdParametersHandler()
        )
        self.root_link_task_param_handler.set_parameter_string(
            name="frame_name", value=self.robot_model.base_link
        )
        self.root_link_task_param_handler.set_parameter_float(
            name="kp_angular", value=tsid_parameters.root_link_kp_ang
        )
        self.root_link_task_param_handler.set_parameter_float(
            name="kd_angular", value=tsid_parameters.root_link_kd_ang
        )
        self.root_link_task_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.root_link_task.set_kin_dyn(self.kindyn)
        self.root_link_task.initialize(param_handler=self.root_link_task_param_handler)

        self.left_foot_regularization_task = blf.tsid.VariableRegularizationTask()
        self.left_foot_regularization_task_name = "left_foot_regularization_task"
        self.left_foot_regularization_task_priority = 1
        self.left_foot_regularization_task_weight = 1e-1 * np.ones(6)
        self.left_foot_regularization_task_param_handler = (
            blf.parameters_handler.StdParametersHandler()
        )
        self.left_foot_regularization_task_param_handler.set_parameter_string(
            name="variable_name", value=self.variable_name_left_contact
        )
        self.left_foot_regularization_task_param_handler.set_parameter_int(
            name="variable_size", value=6
        )
        self.left_foot_regularization_task.initialize(
            param_handler=self.left_foot_regularization_task_param_handler
        )

        self.right_foot_regularization_task = blf.tsid.VariableRegularizationTask()
        self.right_foot_regularization_task_name = "right_foot_regularization_task"
        self.right_foot_regularization_task_priority = 1
        self.right_foot_regularization_task_weight = 1e-1 * np.ones(6)
        self.right_foot_regularization_task_param_handler = (
            blf.parameters_handler.StdParametersHandler()
        )
        self.right_foot_regularization_task_param_handler.set_parameter_string(
            name="variable_name", value=self.variable_name_right_contact
        )
        self.right_foot_regularization_task_param_handler.set_parameter_int(
            name="variable_size", value=6
        )
        self.right_foot_regularization_task.initialize(
            param_handler=self.right_foot_regularization_task_param_handler
        )

        self.angular_momentum_task = blf.tsid.AngularMomentumTask()
        self.angular_momentum_task_name = "angular_momentum_task"
        self.angular_momentum_task_priority = 1
        self.angular_momentum_task_weight = 1e5 * np.ones(3)
        self.angular_momentum_task_parameter_handler = (
            blf.parameters_handler.StdParametersHandler()
        )
        self.angular_momentum_task_parameter_handler.set_group(
            "CONTACT_0", contact_group_left
        )
        self.angular_momentum_task_parameter_handler.set_group(
            "CONTACT_1", contact_group_right
        )
        self.angular_momentum_task_parameter_handler.set_parameter_float(
            name="kp", value=10.0
        )
        self.angular_momentum_task.set_kin_dyn(self.kindyn)
        self.angular_momentum_task.initialize(
            self.angular_momentum_task_parameter_handler
        )

        self.controller.add_task(
            self.joint_tracking_task,
            self.joint_tracking_task_name,
            self.joint_tracking_task_priority,
            self.joint_tracking_task_weight,
        )
        self.controller.add_task(
            self.joint_dynamics_task,
            self.joint_dynamics_task_name,
            self.joint_dynamics_task_priority,
            self.joint_dynamics_task_weight,
        )
        self.controller.add_task(
            self.CoM_Task,
            self.CoM_task_name,
            self.CoM_task_priority,
            self.CoM_task_weight,
        )
        self.controller.add_task(
            self.left_foot_tracking_task,
            self.left_foot_tracking_task_name,
            self.left_foot_tracking_task_priority,
        )
        self.controller.add_task(
            self.right_foot_tracking_task,
            self.right_foot_tracking_task_name,
            self.right_foot_tracking_task_priority,
        )
        self.controller.add_task(
            self.base_dynamic_task,
            self.base_dynamic_task_name,
            self.base_dynamic_task_priority,
            self.base_dynamic_task_weight,
        )
        self.controller.add_task(
            self.root_link_task,
            self.root_link_task_name,
            self.root_link_task_priority,
            self.root_link_task_weight,
        )
        self.controller.add_task(
            self.left_foot_regularization_task,
            self.left_foot_regularization_task_name,
            self.left_foot_regularization_task_priority,
            self.left_foot_regularization_task_weight,
        )
        self.controller.add_task(
            self.right_foot_regularization_task,
            self.right_foot_regularization_task_name,
            self.right_foot_regularization_task_priority,
            self.right_foot_regularization_task_weight,
        )
        self.controller.add_task(
            self.left_foot_wrench_task,
            self.left_foot_wrench_task_name,
            self.left_foot_wrench_priority,
            self.left_foot_wrench_task_weight,
        )
        self.controller.add_task(
            self.right_foot_wrench_task,
            self.right_foot_wrench_task_name,
            self.right_foot_wrench_priority,
            self.right_foot_wrench_task_weight,
        )
        self.controller.add_task(
            self.angular_momentum_task,
            self.angular_momentum_task_name,
            self.angular_momentum_task_priority,
            self.angular_momentum_task_weight,
        )
        self.controller.finalize(self.var_handler)

    def run(self):
        controller_succeeded = self.controller.advance()
        self.torque = self.controller.get_output().joint_torques
        return controller_succeeded

    def update_tasks(
        self,
        s_con_des,
        com_des,
        left_wrench_des,
        right_wrench_des,
        left_foot_pose_des,
        left_foot_mixed_vel_des,
        left_foot_mixed_acc_des,
        left_foot_in_contact,
        right_foot_pose_des,
        right_foot_mixed_vel_des,
        right_foot_mixed_acc_des,
        right_foot_in_contact,
        ang_mom_des,
        root_ori_des,
    ):
        self.CoM_Task.set_set_point(com_des[:3], com_des[3:6], com_des[6:])
        self.left_foot_tracking_task.set_set_point(
            left_foot_pose_des, left_foot_mixed_vel_des, left_foot_mixed_acc_des
        )
        self.right_foot_tracking_task.set_set_point(
            right_foot_pose_des, right_foot_mixed_vel_des, right_foot_mixed_acc_des
        )
        self.update_contacts(left_foot_in_contact, right_foot_in_contact)

        self.joint_tracking_task.set_set_point(s_con_des)
        self.left_foot_regularization_task.set_set_point(left_wrench_des)
        self.right_foot_regularization_task.set_set_point(right_wrench_des)

        self.angular_momentum_task.set_set_point(ang_mom_des, np.zeros(3))

    def update_contacts(self, left_contact, right_contact):
        activate_left_foot_tracking_task = blf.tsid.SE3Task.Enable
        activate_right_foot_tracking_task = blf.tsid.SE3Task.Enable

        if left_contact:
            activate_left_foot_tracking_task = blf.tsid.SE3Task.Disable
        if right_contact:
            activate_right_foot_tracking_task = blf.tsid.SE3Task.Disable

        self.controller.get_task(self.left_foot_wrench_task_name).set_contact_active(
            left_contact
        )
        self.controller.get_task(self.right_foot_wrench_task_name).set_contact_active(
            right_contact
        )
        self.controller.get_task(
            self.right_foot_tracking_task_name
        ).set_task_controller_mode(activate_right_foot_tracking_task)
        self.controller.get_task(
            self.left_foot_tracking_task_name
        ).set_task_controller_mode(activate_left_foot_tracking_task)

    def OutputJointTorque(self, context, output):
        output.SetFromVector(context.get_discrete_state(self.tau_state).value())

    def UpdateOutputJointTorque(self, context, discrete_state):
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

        H_b = RigidTransform(base_quat_wxyz_norm, base_pos)
        s_con = self.qRemap(s_sim, "sim")
        w_b = np.concatenate((base_lin_vel, base_ang_vel))
        ds_con = self.qRemap(ds_sim, "sim")

        self.kindyn.setRobotState(H_b.GetAsMatrix4(), s_con, w_b, ds_con, self.gravity)

        s_sim_des = self.GetInputPort("joint_angle_desired").Eval(context)
        s_con_des = self.qRemap(s_sim_des, "sim")

        com_des_qv = self.GetInputPort("com_desired").Eval(context)
        com_des = np.concatenate((com_des_qv, np.zeros(3)))

        left_wrench_des = self.GetInputPort("left_wrench_desired").Eval(context)
        left_wrench_des = np.concatenate((left_wrench_des, np.zeros(3)))
        right_wrench_des = self.GetInputPort("right_wrench_desired").Eval(context)
        right_wrench_des = np.concatenate((right_wrench_des, np.zeros(3)))

        left_foot_pose_vec = self.GetInputPort("left_foot_pose_desired").Eval(context)

        l_quat_xyzw = np.append(left_foot_pose_vec[1:4], left_foot_pose_vec[0])
        left_foot_pose_des = manif.SE3(left_foot_pose_vec[4:7], l_quat_xyzw)
        left_foot_mixed_vel_des = manif.SE3Tangent(left_foot_pose_vec[7:13])
        left_foot_mixed_acc_des = manif.SE3Tangent(left_foot_pose_vec[13:19])
        left_foot_in_contact = left_foot_pose_vec[19]

        right_foot_pose_vec = self.GetInputPort("right_foot_pose_desired").Eval(context)
        r_quat_xyzw = np.append(right_foot_pose_vec[1:4], right_foot_pose_vec[0])
        right_foot_pose_des = manif.SE3(right_foot_pose_vec[4:7], r_quat_xyzw)
        right_foot_mixed_vel_des = manif.SE3Tangent(right_foot_pose_vec[7:13])
        right_foot_mixed_acc_des = manif.SE3Tangent(right_foot_pose_vec[13:19])
        right_foot_in_contact = right_foot_pose_vec[19]

        root_ori_wxyz = self.GetInputPort("root_ori_desired").Eval(context)
        root_ori_xyzw = np.append(root_ori_wxyz[1:], root_ori_wxyz[0])
        root_ori_des = manif.SO3(root_ori_xyzw)

        ang_mom_des = self.GetInputPort("centroidal_ang_mom_desired").Eval(context)

        self.update_tasks(
            s_con_des,
            com_des,
            left_wrench_des,
            right_wrench_des,
            left_foot_pose_des,
            left_foot_mixed_vel_des,
            left_foot_mixed_acc_des,
            left_foot_in_contact,
            right_foot_pose_des,
            right_foot_mixed_vel_des,
            right_foot_mixed_acc_des,
            right_foot_in_contact,
            ang_mom_des,
            root_ori_des,
        )

        out_flag = self.run()
        tu = self.torque
        self.solve_status = out_flag

        if not out_flag:
            logging.error("QPTSID failed")
            tu = np.zeros(self.na_con)

        u = self.qRemap(tu, "con")
        discrete_state.get_mutable_vector(self.tau_state).set_value(u)

# %%
