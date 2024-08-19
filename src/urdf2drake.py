#  %%
# Imports
from glob import glob
from pathlib import Path
import meshio
from tqdm import tqdm
import xml.etree.ElementTree as ET
from pathlib import Path
import resolve_robotics_uri_py
import itertools
import sys
from amo_urdf import *

# %%
urdf_path = resolve_robotics_uri_py.resolve_robotics_uri(
    "package://ergoCub/robots/ergoCubSN000/model.urdf"
)
mesh_path = urdf_path.__str__().split("ergoCub/robots/ergoCubSN000/model.urdf")[0]
saved_urdf_name = "/model_ergoCub_foot_no_collisions.urdf"

urdf_out_path = Path("./").resolve().__str__() + "/urdfs"

# %%
tree = ET.parse(urdf_path)
root = tree.getroot()

# %%
# read the meshes and convert them from .stl to .obj so that they can be loaded to Drake
print("Converting all the visual meshes")
for child in tqdm(root.findall("./link/visual/geometry/mesh")):
    path = child.attrib["filename"]
    child_mesh_path = mesh_path + path.split("package://")[1]
    child_mesh_name = child_mesh_path.replace("stl", "obj")
    if not Path(child_mesh_name).is_file():
        temp_mesh = meshio.read(child_mesh_path)
        temp_mesh.write(child_mesh_name)
    child.set("filename", path.replace("stl", "obj"))

print("Converting all the collision meshes")
for child in tqdm(root.findall("./link/collision/geometry/mesh")):
    path = child.attrib["filename"]
    child_mesh_path = mesh_path + path.split("package://")[1]
    child_mesh_name = child_mesh_path.replace("stl", "obj")
    if not Path(child_mesh_name).is_file():
        temp_mesh = meshio.read(child_mesh_path)
        temp_mesh.write(child_mesh_name)
    child.set("filename", path.replace("stl", "obj"))

# %%
list = []
for link in root.findall("./link"):
    if link.attrib["name"] not in list:
        for col in link.findall("./collision"):
            link.remove(col)

# %%
red_joint_name_list = [
    "torso_pitch",
    "torso_roll",
    "torso_yaw",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow",
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow",
    "r_hip_pitch",
    "r_hip_roll",
    "r_hip_yaw",
    "r_knee",
    "r_ankle_pitch",
    "r_ankle_roll",
    "l_hip_pitch",
    "l_hip_roll",
    "l_hip_yaw",
    "l_knee",
    "l_ankle_pitch",
    "l_ankle_roll",
]

# %%
joints = root.findall("./joint")
joint_names = []
for j in joints:
    if j.attrib["name"] not in red_joint_name_list:
        j.attrib["type"] = "fixed"

# %%
amo_robot = xml_to_amo(root)
icub_amo = eval(amo_robot)

# %%
# Extract the non-fixed DoF as a list
joints = root.findall("./joint")
joint_names = []
for j in joints:
    if j.attrib["type"] != "fixed":
        joint_names.append([j.attrib["name"], j.attrib["type"]])

# %%
for j in joint_names:
    actuator_name = "actuator_" + j[0]
    transmission_name = "transmission" + j[0]
    temp_trans = Transmission(
        Type("SimpleTransmission"),
        Actuator(Mechanicalreduction("1"), name=actuator_name),
        Transjoint(Hardwareinterface("EffortJointInterface"), j[0]),
        name=transmission_name,
    )
    # Add the transmission to the robot URDF
    icub_amo(temp_trans)

# %%
# Write the modified icub to file
file_name = urdf_out_path + saved_urdf_name
with open(file_name, "w") as f:
    print(icub_amo, file=f)

# %%
print(file_name)

# %%
