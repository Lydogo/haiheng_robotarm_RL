# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))



def object_ee_centering(
    env: ManagerBasedRLEnv,
    std: float,
    threshold_lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    奖励夹爪中心（EE）与物体重心之间的接近程度。
    """
    # 获取物体重心位置 (num_envs, 3)
    object: RigidObject = env.scene[object_cfg.name]
    obj_pos = object.data.root_pos_w

    # 获取夹爪中心（EE）位置 (num_envs, 3)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    # 使用指数函数（高斯核），距离越小奖励越高（最高为1.0）
    dist = torch.norm(obj_pos - ee_pos, dim=1)
    reward=torch.exp(-dist / std)
    current_h = object.data.root_pos_w[:, 2]
    lift_fade = torch.clamp((threshold_lift_height - current_h) / (threshold_lift_height - 0.05), min=0.0, max=1.0)

    return reward * lift_fade


def ee_alignment_to_vector(
    env: ManagerBasedRLEnv,
    std: float,
    threshold_dist: float,
    threshold_lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    EE的X轴与书本长轴向量平行。仅在距离小于threshold_dist时触发随高度平滑衰减，举起后不再强求对齐。
    """
    # 获取书本长轴向量
    object: RigidObject = env.scene[object_cfg.name]
    book_long_axis = quat_apply(object.data.root_quat_w, 
                                torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1))
    book_long_axis = torch.nn.functional.normalize(book_long_axis, dim=1)

    current_h = object.data.root_pos_w[:, 2]
    lift_fade = torch.clamp((threshold_lift_height - current_h) / (threshold_lift_height - 0.05), min=0.0, max=1.0)

    # 获取EE的X轴
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    ee_x_axis = quat_apply(ee_quat, torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1))
    ee_x_axis = torch.nn.functional.normalize(ee_x_axis, dim=1)

    # 计算平行度：点积绝对值（越接近1越平行）
    parallel_error = torch.abs(torch.sum(book_long_axis * ee_x_axis, dim=1))
    reward = torch.exp(-(1.0 - parallel_error) / std)

    # 距离触发逻辑
    dist = torch.norm(ee_frame.data.target_pos_w[..., 0, :] - object.data.root_pos_w, dim=1)
    return (dist < threshold_dist).float() * lift_fade * reward


def ee_perpendicular_to_vector(
    env: ManagerBasedRLEnv,
    std: float,
    threshold_dist: float,
    threshold_lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    EE的Z轴与书本长轴向量垂直。仅在距离小于threshold_dist时触发。
    """
    object: RigidObject = env.scene[object_cfg.name]
    world_down = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    
    # 获取EE的Z轴
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    ee_z_axis = quat_apply(ee_quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1))

    dot_prod = torch.sum(ee_z_axis * world_down, dim=1)
    reward = torch.exp(-(1.0 - dot_prod) / std)

    current_h = object.data.root_pos_w[:, 2]
    lift_fade = torch.clamp((threshold_lift_height - current_h) / (threshold_lift_height - 0.05), min=0.0, max=1.0)

    # 距离触发逻辑
    dist = torch.norm(ee_frame.data.target_pos_w[..., 0, :] - object.data.root_pos_w, dim=1)
    return (dist < threshold_dist).float() * lift_fade * reward