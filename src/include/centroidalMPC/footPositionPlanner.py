import manifpy as manif
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import bipedal_locomotion_framework.bindings as blf
from datetime import timedelta
from IPython.display import clear_output


class FootPositionPlanner:
    def __init__(self, robot_model, dT):
        self.robot_model = robot_model
        self.define_kindyn_functions()
        self.dT = dT

    def define_kindyn_functions(self):
        self.H_left_foot_fun = self.robot_model.forward_kinematics_fun(
            self.robot_model.left_foot_frame
        )
        self.H_right_foot_fun = self.robot_model.forward_kinematics_fun(
            self.robot_model.right_foot_frame
        )

    def update_initial_position(self, Hb, s):
        self.H_left_foot_init = self.H_left_foot_fun(Hb, s)
        self.H_right_foot_init = self.H_right_foot_fun(Hb, s)

    def set_scaling_parameters(self, scaling, scalingPos, scalingPosY):
        self.scaling = scaling
        self.scaling_pos = scalingPos
        self.scaling_pos_y = scalingPosY

    def compute_feet_contact_position(self):
        foot_quaternion_xyzw = [0, 0, 0, 1]
        quaternion = foot_quaternion_xyzw
        self.contact_list_left_foot = blf.contacts.ContactList()
        contact = blf.contacts.PlannedContact()
        leftPosition = np.zeros(3)
        leftPosition_casadi = np.array(self.H_left_foot_init[:3, 3])
        leftPosition[0] = float(leftPosition_casadi[0])
        leftPosition[1] = float(leftPosition_casadi[1])
        leftPosition[2] = float(leftPosition_casadi[2])
        leftPosition[2] = 0.0
        contact.pose = manif.SE3(position=leftPosition, quaternion=quaternion)
        contact.activation_time = timedelta(seconds=0.0)
        contact.deactivation_time = timedelta(seconds=1.0 * self.scaling)
        contact.name = "contactLeft1"
        self.contact_list_left_foot.add_contact(contact)

        leftPosition[0] += float(0.05 * self.scaling_pos)
        contact.pose = manif.SE3(position=leftPosition, quaternion=quaternion)
        contact.activation_time = timedelta(seconds=2.0 * self.scaling)
        contact.deactivation_time = timedelta(seconds=5.0 * self.scaling)
        contact.name = "contactLeft2"
        self.contact_list_left_foot.add_contact(contact)

        leftPosition[0] += 0.1 * self.scaling_pos
        leftPosition[2] = 0.0
        contact.pose = manif.SE3(position=leftPosition, quaternion=quaternion)
        contact.activation_time = timedelta(seconds=6.0 * self.scaling)
        contact.deactivation_time = timedelta(seconds=9.0 * self.scaling)
        contact.name = "contactLeft3"
        self.contact_list_left_foot.add_contact(contact)

        leftPosition[0] += 0.1 * self.scaling_pos
        leftPosition[2] = 0.0
        contact.pose = manif.SE3(position=leftPosition, quaternion=quaternion)
        contact.activation_time = timedelta(seconds=10.0 * self.scaling)
        contact.deactivation_time = timedelta(seconds=13.0 * self.scaling)
        contact.name = "contactLeft4"
        self.contact_list_left_foot.add_contact(contact)

        leftPosition[0] += 0.1 * self.scaling_pos
        leftPosition[2] = 0.0
        contact.pose = manif.SE3(position=leftPosition, quaternion=quaternion)
        contact.activation_time = timedelta(seconds=14.0 * self.scaling)
        contact.deactivation_time = timedelta(seconds=25.0 * self.scaling)
        contact.name = "contactLeft5"
        self.contact_list_left_foot.add_contact(contact)

        self.contact_list_right_foot = blf.contacts.ContactList()
        contact = blf.contacts.PlannedContact()
        rightPosition = np.zeros(3)
        rightPosition_casadi = np.array(self.H_right_foot_init[:3, 3])
        rightPosition[0] = float(rightPosition_casadi[0])
        rightPosition[1] = float(rightPosition_casadi[1])
        rightPosition[2] = float(rightPosition_casadi[2])
        rightPosition[2] = 0.0
        contact.pose = manif.SE3(position=rightPosition, quaternion=quaternion)
        contact.activation_time = timedelta(seconds=0.0)
        contact.deactivation_time = timedelta(seconds=3.0 * self.scaling)
        contact.name = "contactRight1"
        self.contact_list_right_foot.add_contact(contact)

        rightPosition[0] += 0.1 * self.scaling_pos
        contact.pose = manif.SE3(position=rightPosition, quaternion=quaternion)
        contact.activation_time = timedelta(seconds=4.0 * self.scaling)
        contact.deactivation_time = timedelta(seconds=7.0 * self.scaling)
        contact.name = "contactRight2"
        self.contact_list_right_foot.add_contact(contact)

        rightPosition[0] += 0.1 * self.scaling_pos
        contact.pose = manif.SE3(position=rightPosition, quaternion=quaternion)
        contact.activation_time = timedelta(seconds=8.0 * self.scaling)
        contact.deactivation_time = timedelta(seconds=11.0 * self.scaling)
        contact.name = "contactRight3"
        self.contact_list_right_foot.add_contact(contact)

        rightPosition[0] += 0.1 * self.scaling_pos
        contact.pose = manif.SE3(position=rightPosition, quaternion=quaternion)
        contact.activation_time = timedelta(seconds=12.0 * self.scaling)
        contact.deactivation_time = timedelta(seconds=15.0 * self.scaling)
        contact.name = "contactRight4"
        self.contact_list_right_foot.add_contact(contact)

        rightPosition[0] += 0.05 * self.scaling_pos
        contact.pose = manif.SE3(position=rightPosition, quaternion=quaternion)
        contact.activation_time = timedelta(seconds=16.0 * self.scaling)
        contact.deactivation_time = timedelta(seconds=25.0 * self.scaling)
        contact.name = "contactRight5"
        self.contact_list_right_foot.add_contact(contact)

    def get_contact_phase_list(self):
        contact_list_map = {}
        contact_list_map.update({"left_foot": self.contact_list_left_foot})
        contact_list_map.update({"right_foot": self.contact_list_right_foot})
        contact_phase_list = blf.contacts.ContactPhaseList()
        contact_phase_list.set_lists(contact_list_map)
        return contact_phase_list

    def plot_feet_position(self):
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        for item in self.contact_list_left_foot:
            pos = item.pose.translation()
            plt.plot(pos[0], pos[1], marker="D", color="red")
            ax.add_patch(
                Rectangle(
                    (pos[0] - 0.015, pos[1] - 0.01), 0.03, 0.02, color="red", alpha=0.2
                )
            )
            ax.text(pos[0], pos[1], str(item.activation_time), style="italic")
        for item in self.contact_list_right_foot:
            pos = item.pose.translation()
            plt.plot(pos[0], pos[1], marker="D", color="blue")
            ax.add_patch(
                Rectangle(
                    (pos[0] - 0.015, pos[1] - 0.01), 0.03, 0.02, color="blue", alpha=0.2
                )
            )
            ax.text(pos[0], pos[1], str(item.activation_time), style="italic")
        plt.title("Feet Position")
        plt.xlabel(
            "x [m]",
        )
        plt.ylabel(
            "y [m]",
        )
        plt.show()

    def initialize_foot_swing_planner(self):
        self.parameters_handler = blf.parameters_handler.StdParametersHandler()
        self.parameters_handler.set_parameter_datetime("sampling_time", self.dT)
        self.parameters_handler.set_parameter_float("step_height", 0.03)
        self.parameters_handler.set_parameter_float("foot_apex_time", 0.5)
        self.parameters_handler.set_parameter_float("foot_landing_velocity", 0.0)
        self.parameters_handler.set_parameter_float("foot_landing_acceleration", 0.0)
        self.parameters_handler.set_parameter_float("foot_take_off_velocity", 0.0)
        self.parameters_handler.set_parameter_float("foot_take_off_acceleration", 0.0)
        self.parameters_handler.set_parameter_string(
            "interpolation_method", "min_acceleration"
        )

        self.planner_left_foot = blf.planners.SwingFootPlanner()
        self.planner_right_foot = blf.planners.SwingFootPlanner()
        self.planner_left_foot.initialize(handler=self.parameters_handler)
        self.planner_right_foot.initialize(handler=self.parameters_handler)
        self.planner_left_foot.set_contact_list(
            contact_list=self.contact_list_left_foot
        )
        self.planner_right_foot.set_contact_list(
            contact_list=self.contact_list_right_foot
        )

    def advance_swing_foot_planner(self):
        self.planner_left_foot.advance()
        self.planner_right_foot.advance()

    def get_references_swing_foot_planner(self):
        left_foot = self.planner_left_foot.get_output()
        right_foot = self.planner_right_foot.get_output()
        return left_foot, right_foot

    def plot_foot_trajectories(self):
        T = 0
        lfoot_ref = []
        rfoot_ref = []
        while T < 30:
            self.advance_swing_foot_planner()
            lfoot, rfoot = self.get_references_swing_foot_planner()
            lfoot_ref.append(lfoot.transform.translation())
            rfoot_ref.append(rfoot.transform.translation())
            T += self.dT
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        left_data = np.array(lfoot_ref)
        right_data = np.array(rfoot_ref)
        ax.scatter(
            left_data[:, 0],
            left_data[:, 1],
            left_data[:, 2],
            color="green",
            label="left_foot",
        )
        ax.scatter(
            right_data[:, 0],
            right_data[:, 1],
            right_data[:, 2],
            color="blue",
            label="right_foot",
        )
        plt.legend()
        plt.show()
