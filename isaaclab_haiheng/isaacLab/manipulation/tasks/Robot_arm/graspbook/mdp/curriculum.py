from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def modify_reward_weight_linear(
    env: ManagerBasedRLEnv,
    env_ids: list[int],
    term_name: str,
    initial_weight: float,
    target_weight: float,
    num_steps_start: int,
    num_steps_stop: int,
) -> float:
    """在指定的步数区间内，线性地将奖励权重从初始值修改为目标值。"""
    reward_manager = env.reward_manager
    term_cfg = reward_manager.get_term_cfg(term_name)
    current_step = env.common_step_counter

    if current_step <= num_steps_start:
        pass 
    elif current_step >= num_steps_stop:
        # 超过结束步数，固定为目标权重
        term_cfg.weight = target_weight
    else:
        # 在区间内，进行线性计算,计算进度比例 (0.0 ~ 1.0)
        progress = (current_step - num_steps_start) / (num_steps_stop - num_steps_start)
        new_weight = initial_weight + progress * (target_weight - initial_weight)
        term_cfg.weight = new_weight

    reward_manager.set_term_cfg(term_name, term_cfg)
    return term_cfg.weight