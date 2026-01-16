# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg,MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaacLab.manipulation.tasks.Robot_arm.graspbook import mdp
from isaacLab.manipulation.tasks.Robot_arm.graspbook.graspbook_env_cfg import GraspBookEnvCfg

# 用于生成基础几何体
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaacLab.manipulation.assets.config.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaGraspBookEnvCfg(GraspBookEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"


        stand_size = (0.2, 0.3, 0.5) # 长0.2m, 宽0.3m, 高0.5m
        self.scene.stand = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Stand",
            spawn=CuboidCfg(
                size=stand_size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)), # 灰色台子
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # 将台子设为静态
                rigid_props=RigidBodyPropertiesCfg(
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=1000.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.8, 0, stand_size[2] / 2],rot=[1, 0, 0, 0]),
        )

        # Set Book as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.11], rot=[1, 0, 0, 0]),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.8, 0, 0.61], rot=[1, 0, 0, 0]),
            spawn=CuboidCfg(
                size=(0.14, 0.05, 0.20),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.2, 0.5), # 书籍封面色（深蓝） 
                    roughness=0.5
                ),
                mass_props=MassPropertiesCfg(mass=0.2),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                # 添加碰撞属性
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
        )


        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaGraspBookEnvCfg_PLAY(FrankaGraspBookEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
