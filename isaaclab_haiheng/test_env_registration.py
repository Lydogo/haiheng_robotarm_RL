#!/usr/bin/env python3
"""
测试环境注册的脚本
"""

import gymnasium as gym

def test_env_registration():
    """测试环境注册"""
    print("正在检查已注册的环境...")
    
    # 尝试导入我们的包
    try:
        import isaacLab.manipulation
        print("✓ 成功导入 isaacLab.manipulation")
    except Exception as e:
        print(f"✗ 导入 isaacLab.manipulation 失败: {e}")
        return
    
    # 检查已注册的环境
    isaac_envs = []
    for env_id in gym.registry.keys():
        if "Isaac" in env_id:
            isaac_envs.append(env_id)
    
    if isaac_envs:
        print(f"✓ 找到 {len(isaac_envs)} 个 Isaac 环境:")
        for env_id in sorted(isaac_envs):
            print(f"  - {env_id}")
    else:
        print("✗ 没有找到任何 Isaac 环境")
    
    # 检查特定的环境
    specific_envs = [
        "Template-Isaac-Lift-Cube-Franka-v0",
        "Template-Isaac-Lift-Cube-Franka-IK-Abs-v0",
        "Template-Isaac-Reach-Franka-v0"
    ]
    
    print("\n检查特定环境:")
    for env_id in specific_envs:
        if env_id in gym.registry:
            print(f"✓ {env_id} 已注册")
        else:
            print(f"✗ {env_id} 未注册")

if __name__ == "__main__":
    test_env_registration() 