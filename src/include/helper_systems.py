import numpy as np

from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.plant import ExternallyAppliedSpatialForce_
from pydrake.systems.framework import BasicVector, LeafSystem


# %%
class helper_class(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareVectorOutputPort("foot_pose", BasicVector(7), self.calcOutput)

    def calcOutput(self, context, output):
        foot_idx = self.plant.GetBodyByName("l_sole").index()
        foot_pose_T = self.GetInputPort("body_poses").Eval(context)[foot_idx]
        foot_pose = np.concatenate(
            (foot_pose_T.rotation().ToQuaternion().wxyz(), foot_pose_T.translation())
        )
        output.SetFromVector(foot_pose)


# %%
