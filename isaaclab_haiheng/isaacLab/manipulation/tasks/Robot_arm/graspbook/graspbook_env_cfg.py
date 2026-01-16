# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as general_mdp
##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    # 在桌子上放置一个台子，模拟书架将书抬高
    stand: RigidObjectCfg = MISSING
    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = general_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=general_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.65, 0.68), pos_y=(-0.05, 0.05), pos_z=(0.58, 0.63), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: general_mdp.JointPositionActionCfg | general_mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: general_mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=general_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=general_mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_orientation_vector = ObsTerm(func=mdp.object_long_axis_vector)
        target_object_position = ObsTerm(func=general_mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=general_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=general_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0),"yaw": (-0.785, 0.785)},
            "pose_range": {"x": (0.0, 0.0), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=2.5,
    )

    touching_object = RewTerm(
        func=mdp.object_ee_centering,
        params={"std": 0.08,"extraction_threshold":0.75},
        weight=2.0,
    )
    
    # 引导平行奖励
    ee_x_parallel_book = RewTerm(
        func=mdp.ee_alignment_to_vector,
        params={"std": 0.08,"threshold_dist": 0.08},
        weight=1.8,
    )

    pulling_object = RewTerm(
        func=mdp.object_is_pulled,
        params={"extraction_threshold": 0.72,"std_align":0.2,"ori_influence":0.3},
        weight=28.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "extraction_threshold": 0.72, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "extraction_threshold": 0.72, "command_name": "object_pose"},
        weight=10.0,
    )


    # action penalty
    action_rate = RewTerm(
        func=general_mdp.action_rate_l2,
        weight=-1e-4,)

    joint_vel = RewTerm(
        func=general_mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-1e-4,
    )

    object_fallen_penalty = RewTerm(
        func=mdp.object_fallen, 
        params={"object_cfg": SceneEntityCfg("object"), "minimum_height": 0.53},
        weight=-2.0,
)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=general_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=general_mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    object_fallen = DoneTerm(
        func=mdp.object_fallen,
        params={"object_cfg": SceneEntityCfg("object"), "minimum_height": 0.53}
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    action_rate = CurrTerm(
        func=general_mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.4, "num_steps": 8000}
    )

    joint_vel = CurrTerm(
        func=general_mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.4, "num_steps": 8000}
    )

    # smooth_action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight_linear,
    #     params={
    #         "term_name": "action_rate",
    #         "initial_weight": -1e-4,
    #         "target_weight": -0.2,
    #         "num_steps_start": 1200,
    #         "num_steps_stop": 2000,
    #     }
    # )

    # smooth_joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight_linear,
    #     params={
    #         "term_name": "joint_vel",
    #         "initial_weight": -1e-4,
    #         "target_weight": -0.1,
    #         "num_steps_start": 1000,
    #         "num_steps_stop": 2000,
    #     }
    # )

##
# Environment configuration
##


@configclass
class GraspBookEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the grasp environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
