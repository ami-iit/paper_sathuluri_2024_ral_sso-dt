
# %%
import logging

import numpy as np
from pydrake.common.value import Value

from pydrake.multibody.math import SpatialForce
from pydrake.multibody.plant import ExternallyAppliedSpatialForce_
from pydrake.systems.framework import BasicVector, LeafSystem
from pydrake.geometry import Rgba, Cylinder
from pydrake.math import RigidTransform, RotationMatrix


# %%

class JointActuation(LeafSystem):
    def __init__(
        self,
        plant,
        K_N,
        motor_inertia_matrix,
        tau_sat,
        f_K0,
        f_K1,
        f_K2,
        K_bemf,
        delay=0.0,
    ):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort(
            "demanded_torque", BasicVector(plant.num_actuators())
        )
        self.DeclareVectorInputPort(
            "robot_state", BasicVector(plant.num_positions() + plant.num_velocities())
        )
        self.DeclareVectorInputPort(
            "generalised_accelerations", BasicVector(plant.num_velocities())
        )
        self.DeclareVectorOutputPort(
            "realised_torque", BasicVector(plant.num_actuators()), self.SetOutput
        )
        self.DeclareVectorOutputPort(
            "friction_torque",
            BasicVector(plant.num_actuators()),
            self.GetFrictionTorque,
        )
        self.DeclareVectorOutputPort(
            "motor_torque",
            BasicVector(plant.num_actuators()),
            self.GetMotorTorque,
        )
        self.DeclareVectorOutputPort(
            "motor_velocity",
            BasicVector(plant.num_actuators()),
            self.GetMotorVelocity,
        )
        self.dT = delay

        if self.dT != 0.0:
            self.jtau_state = self.DeclareDiscreteState(plant.num_actuators())
            self.DeclarePeriodicDiscreteUpdateEvent(
                period_sec=self.dT, offset_sec=0.0, update=self.UpdateOutputJointTorque
            )
            logging.info("Updating ActuationSystem at: {} Hz".format(1 / self.dT))
        else:
            logging.info("Updating ActuationSystem at: Continuous")

        self.plant = plant
        self.K_N = K_N
        self.f_K0 = f_K0
        self.f_K1 = f_K1
        self.f_K2 = f_K2
        self.K_bemf = K_bemf
        self.tau_sat = tau_sat
        self.I_m = motor_inertia_matrix
        self.nq = self.plant.num_positions()
        self.nv = self.plant.num_velocities()
        self.na = self.plant.num_actuators()

        self.friction_torque = np.zeros(self.na)
        self.motor_torque = np.zeros(self.na)
        self.motor_velocity = np.zeros(self.na)

    def SetOutput(self, context, output):
        if self.dT != 0.0:
            output.SetFromVector(context.get_discrete_state(self.jtau_state).value())
        else:
            output.SetFromVector(self.ComputeJointTorque(context))

    def GetFrictionTorque(self, context, output):
        output.SetFromVector(self.friction_torque)

    def GetMotorTorque(self, context, output):
        output.SetFromVector(self.motor_torque)

    def GetMotorVelocity(self, context, output):
        output.SetFromVector(self.motor_velocity)

    def ComputeJointTorque(self, context):
        robot_state = self.GetInputPort("robot_state").Eval(context)
        generalised_acceleration = self.GetInputPort("generalised_accelerations").Eval(
            context
        )
        joint_velocity = robot_state[self.nq + 6 :]
        self.motor_velocity = self.K_N.dot(joint_velocity)

        joint_acceleration = generalised_acceleration[6:]
        demanded_torque = self.GetInputPort("demanded_torque").Eval(context)
        demanded_torque_sat = np.clip(demanded_torque, -self.tau_sat, self.tau_sat)
        self.motor_torque = demanded_torque_sat / np.diag(self.K_N)

        inertia_torque = self.K_N.dot(self.K_N.dot(self.I_m.dot(joint_acceleration)))
        friction_torque = self.f_K0.dot(
            np.tanh(self.f_K1.dot(joint_velocity))
        ) + self.f_K2.dot(joint_velocity)
        self.friction_torque = friction_torque
        bemf_torque = self.K_N.dot(
            self.K_N.dot(self.K_bemf.dot(np.abs(joint_velocity)))
        )
        realised_torque = (
            demanded_torque_sat - inertia_torque - friction_torque - bemf_torque
        )
        return realised_torque

    def UpdateOutputJointTorque(self, context, discrete_state):
        realised_torque = self.ComputeJointTorque(context)
        discrete_state.get_mutable_vector(self.jtau_state).set_value(realised_torque)


# %%


class DisturbanceSystem(LeafSystem):
    def __init__(
        self,
        plant,
        body_name,
        rng,
        force_type,
        force_params,
        time,
        duration,
        meshcat=None,
    ):
        LeafSystem.__init__(self)
        self.meshcat = meshcat
        self.nv = plant.num_velocities()
        self.body_name = body_name
        self.target_body_index = plant.GetBodyByName(body_name).index()
        self.plant = plant
        self.t = time
        self.dur = duration
        forces_cls = Value[list[ExternallyAppliedSpatialForce_[float]]]
        self.apply_force = False
        self.start_force_application_time = 0.0
        self.force_type = force_type
        self.force_params = force_params
        self.nmin = self.force_params["min"]
        self.nmax = self.force_params["max"]
        self.rng = rng
        self.random_wrench = np.zeros(6)

        self.DeclareAbstractOutputPort(
            "spatial_forces_vector",
            lambda: forces_cls(),
            self.DoCalcAbstractOutput,
        )

    def DoCalcAbstractOutput(self, context, y_data):
        test_force = ExternallyAppliedSpatialForce_[float]()
        test_force.body_index = self.target_body_index
        test_force.p_BoBq_B = np.zeros(3)
        marker_path = f"/drake/visualizer/ergoCub/{self.body_name}/disturbance"
        marker_scale = 1e-2
        test_force.F_Bq_W = SpatialForce(
            tau=np.array([0.0, 0.0, 0.0]), f=np.array([0.0, 0.0, 0.0])
        )
        current_time = context.get_time()
        if current_time % self.t == 0 and current_time != 0:
            self.apply_force = True
            self.start_force_application_time = current_time
            if self.force_type == "uniform":
                self.random_wrench = self.rng.uniform(self.nmin, self.nmax)
        if self.apply_force:
            test_force.F_Bq_W = SpatialForce(
                tau=self.random_wrench[0:3], f=self.random_wrench[3:6]
            )
            if self.meshcat is not None:
                force_magnitude = np.linalg.norm(self.random_wrench[3:6])
                unit_vector = self.random_wrench[3:6] / force_magnitude
                self.meshcat.SetObject(
                    path=marker_path,
                    shape=Cylinder(radius=0.025, length=force_magnitude * marker_scale),
                    rgba=Rgba(1, 0, 0, 1),
                )
                transform = RigidTransform(
                    RotationMatrix.MakeFromOneVector(unit_vector, 2),
                )
                self.meshcat.SetTransform(path=marker_path, X_ParentPath=transform)
            if current_time - self.start_force_application_time >= self.dur:
                self.apply_force = False
                if self.meshcat is not None:
                    self.meshcat.Delete(marker_path)

        y_data.set_value([test_force])


# %%
class NoiseAdder(LeafSystem):
    def __init__(
        self,
        plant,
        size,
        idx=None,
        rng=None,
        noise_type="gaussian",
        noise_params={"min": np.array([0]), "max": np.array([1])},
    ):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("signal", BasicVector(size))
        self.DeclareVectorOutputPort(
            "noisy_signal",
            BasicVector(size),
            self.DoCalcOutput,
        )
        self.rng = rng
        self.noise_type = noise_type
        self.noise_params = noise_params
        if idx is None:
            self.idx = np.arange(0, size)
        else:
            self.idx = idx

    def DoCalcOutput(self, context, output):
        signal = self.get_input_port(0).Eval(context)
        nmin = self.noise_params["min"]
        nmax = self.noise_params["max"]
        rn = []
        if self.noise_type == "gaussian":
            for ii in range(len(nmax)):
                rn.append(nmin[ii] + (nmax[ii] - nmin[ii]) * self.rng.normal(0, 1, 1))
        if self.noise_type == "uniform":
            for ii in range(len(nmax)):
                rn.append(nmin[ii] + (nmax[ii] - nmin[ii]) * self.rng.uniform(0, 1, 1))
        if self.noise_type == "max":
            for ii in range(len(nmax)):
                rn.append(nmax[ii])
        signal[self.idx] += np.array(rn).flatten()

        output.SetFromVector(signal)
