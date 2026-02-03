# haiheng_robotarm_RL

## Version 26.1.9
根据Isaaclab模板，使用franka机械臂进行自定义抓取训练

## 1.docker运行
由于该主机为20.04，无法使用pip安装isaacsim，因此直接使用官方教程中的docker部署方式(详细请参考Isaaclab官方文档)
Isaaclab源码保存在：/workspace/isaaclab/source/isaaclab_haiheng/isaacLab.manipulation
```bash
# docker启动命令
~/IsaacLab$ ./docker/container.py start
~/IsaacLab$ ./docker/container.py enter
```
本项目采用以下开源库作为模板
https://github.com/NathanWu7/isaacLab.manipulation/tree/main/isaacLab/manipulation
```bash
# 训练指令
/workspace/isaaclab# python3 scripts/rsl_rl/train.py --task Grasp-Book-Franka-v0 --num_envs 4096 --headless
/workspace/isaaclab# python3 scripts/rsl_rl/play.py --task Grasp-Book-Franka-v0 --num_envs 1
```


## 2.代码说明
- 新增加任务需要在init里面增加注册表，路径如下：
/workspace/isaaclab/source/isaaclab_haiheng/isaacLab.manipulation/isaacLab/manipulation/tasks/Robot_arm/graspbook/config/franka/__init__.py
- /graspbook/mdp下是mdp的具体策略代码，如奖励函数、终止条件、观测量；而训练的权重等参数需要在/graspbook/graspbook_env_cfg.py里进行修改
- /graspbook/config/franka/joint_pos_env_cfg.py里面可以修改环境配置，如书本、书架、机器人位置等参数
 

