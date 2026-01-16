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


def object_ee_centering(
    env: ManagerBasedRLEnv,
    std: float,
    extraction_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    奖励夹爪中心（EE）与物体重心之间的接近程度，随取出距离平滑衰减。
    """
    initial_x = 0.8
    # 获取物体重心位置 (num_envs, 3)
    object: RigidObject = env.scene[object_cfg.name]
    obj_pos = object.data.root_pos_w
    # 获取夹爪中心（EE）位置 (num_envs, 3)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    # 使用指数函数（高斯核），距离越小奖励越高（最高为1.0）
    dist = torch.norm(obj_pos - ee_pos, dim=1)
    reward=torch.exp(-dist / std)
    current_relative_x = obj_pos[:, 0] - env.scene.env_origins[:, 0]
    extraction_fade = torch.clamp(
        (current_relative_x - extraction_threshold) / (initial_x - extraction_threshold), 
        min=0.0, 
        max=1.0
    )
    return reward * extraction_fade


def ee_alignment_to_vector(
    env: ManagerBasedRLEnv,
    std: float,
    threshold_dist: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    EE的X轴与书本长轴向量平行。仅在距离小于threshold_dist时触发
    """
    # 获取书本长轴向量
    object: RigidObject = env.scene[object_cfg.name]
    book_long_axis = quat_apply(object.data.root_quat_w, 
                                torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1))
    book_long_axis = torch.nn.functional.normalize(book_long_axis, dim=1)
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
    return (dist < threshold_dist).float() * reward


def object_is_pulled(
    env: ManagerBasedRLEnv, 
    extraction_threshold: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    std_align: float = 0.2,
    ori_influence: float = 0.5,
) -> torch.Tensor:
    """奖励智能体将物体从书架中拉出，提供连续的位移引导。"""
    initial_x = 0.8
    object: RigidObject = env.scene[object_cfg.name]
    relative_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    pull_progress = torch.clamp(
        (initial_x - relative_x) / (initial_x - extraction_threshold), 
        min=0.0, 
        max=1.0
    )

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    ee_x_axis = quat_apply(ee_quat, torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1))
    ee_x_axis = torch.nn.functional.normalize(ee_x_axis, dim=1)
    world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    dot_world_z = torch.sum(ee_x_axis * world_z_axis, dim=1)
    world_z_align_reward = torch.exp(-(1.0 - dot_world_z) / std_align)

    orientation_guidance = pull_progress * ((1-ori_influence) + ori_influence * world_z_align_reward)
    is_pulled_bonus = (relative_x < extraction_threshold).float() * world_z_align_reward
    return is_pulled_bonus + orientation_guidance


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    extraction_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    initial_x = 0.8
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    relative_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    pull_progress = torch.clamp((initial_x - relative_x) / (initial_x - extraction_threshold), min=0.0, max=1.0)
    return pull_progress * (1 - torch.tanh(distance / std))


def object_vertical_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    activation_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """奖励物体保持竖直姿态（物体局部Z轴与世界坐标系Z轴平行）。"""
    initial_x = 0.8
    object: RigidObject = env.scene[object_cfg.name]
    quat = object.data.root_quat_w
    object_z_axis_w = quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1))
    world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    dot_product = torch.sum(object_z_axis_w * world_z_axis, dim=1)
    reward = torch.exp(-(1.0 - dot_product) / std)
    relative_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    activation_scale = torch.clamp((initial_x - relative_x) / (initial_x - activation_threshold), min=0.0, max=1.0)
    return activation_scale * reward


def object_fallen(
    env: ManagerBasedRLEnv, 
    minimum_height: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """判定物体是否倾倒或掉落。
    当物体重心高度低于阈值时返回 1.0，否则返回 0.0。
    """
    object: RigidObject = env.scene[object_cfg.name]
    object_z = object.data.root_pos_w[:, 2]
    return (object_z < minimum_height).float()