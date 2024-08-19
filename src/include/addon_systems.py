### Additional drake `systems` that can be added to the robot model externally

# %%
import logging

import numpy as np
from pydrake.common.value import Value
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.plant import ExternallyAppliedSpatialForce_
from pydrake.systems.framework import BasicVector, LeafSystem


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
# Similar to: https://github.com/vincekurtz/valkyrie_drake/blob/master/utils/disturbance_system.py


class DisturbanceSystem(LeafSystem):
    def __init__(self, plant, body_name, spatial_force, time, duration):
        LeafSystem.__init__(self)
        self.nv = plant.num_velocities()
        self.target_body_index = plant.GetBodyByName(body_name).index()
        self.t = time
        self.wrench = spatial_force.reshape(6)  # spatial_force/wrench = [torque; force]
        self.dur = duration
        forces_cls = Value[list[ExternallyAppliedSpatialForce_[float]]]

        self.DeclareAbstractOutputPort(
            "spatial_forces_vector",
            lambda: forces_cls(),
            self.DoCalcAbstractOutput,
        )

    def DoCalcAbstractOutput(self, context, y_data):
        test_force = ExternallyAppliedSpatialForce_[float]()
        test_force.body_index = self.target_body_index
        test_force.p_BoBq_B = np.zeros(3)
        if context.get_time() % self.t == 0 and context.get_time() != 0:
            test_force.F_Bq_W = SpatialForce(tau=self.wrench[0:3], f=self.wrench[3:6])
        else:
            test_force.F_Bq_W = SpatialForce(
                tau=np.array([0.0, 0.0, 0.0]), f=np.array([0.0, 0.0, 0.0])
            )
        y_data.set_value([test_force])


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
        signal[self.idx] += rn
        output.SetFromVector(signal)
