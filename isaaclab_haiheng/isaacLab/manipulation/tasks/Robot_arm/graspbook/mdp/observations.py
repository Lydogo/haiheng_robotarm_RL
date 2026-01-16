# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms,quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


# 用于观测书本的向量
def object_long_axis_vector(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """
    获取书本长轴在世界坐标系下的方向向量。
    对应视觉返回的：从面中心指向长边延伸方向的向量。
    """
    # 获取物体在世界系下的姿态 (四元数)
    object: RigidObject = env.scene[object_cfg.name]
    object_quat = object.data.root_quat_w

    # 获取局部坐标
    local_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    
    # 将局部向量旋转到世界坐标系
    # world_vec = quat * local_vec * quat_inv
    world_axis = quat_apply(object_quat, local_axis)
    return world_axis