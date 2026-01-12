# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a socket-based teleoperation receiver for Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import socket
import json
import threading
from typing import Dict, Any, Optional

import gymnasium as gym
import torch
import numpy as np



from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Socket teleoperation receiver for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--socket_host", type=str, default="localhost", help="Socket host address.")
parser.add_argument("--socket_port", type=int, default=12345, help="Socket port number.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat
import omni.log

def euler_to_quat(euler_angles):
    """
    将欧拉角(roll, pitch, yaw)转换为四元数(w, x, y, z)
    
    参数:
    roll -- 绕X轴的旋转角度(弧度)
    pitch -- 绕Y轴的旋转角度(弧度)
    yaw -- 绕Z轴的旋转角度(弧度)
    
    返回:
    四元数 [w, x, y, z]
    """
    roll, pitch, yaw = euler_angles
    # 计算半角
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    # 计算四元数分量
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return [w, x, y, z]

class SocketCommandReceiver:
    """Socket命令接收器，处理来自controller_mcp.py的控制指令"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.command_queue = []
        self.lock = threading.Lock()
        
    def start_server(self):
        """启动socket服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.running = True
            
            print(f"Socket服务器启动成功，监听 {self.host}:{self.port}")
            
            # 启动接收线程
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
        except Exception as e:
            print(f"Socket服务器启动失败: {e}")
            self.running = False
    
    def _receive_loop(self):
        """接收循环"""
        while self.running:
            try:
                print("等待客户端连接...")
                self.client_socket, addr = self.server_socket.accept()
                print(f"客户端已连接: {addr}")
                
                while self.running and self.client_socket:
                    try:
                        # 接收数据
                        data = self.client_socket.recv(4096)
                        if not data:
                            break
                            
                        # 解析命令
                        command = json.loads(data.decode('utf-8'))
                        print(f"收到命令: {command}")
                        
                        # 将命令添加到队列
                        with self.lock:
                            self.command_queue.append(command)
                            
                        # 不再立即发送响应，让主循环处理响应
                        
                    except json.JSONDecodeError:
                        response = {"success": False, "error": "无效的JSON格式"}
                        self.client_socket.send(json.dumps(response).encode('utf-8'))
                    except Exception as e:
                        response = {"success": False, "error": str(e)}
                        self.client_socket.send(json.dumps(response).encode('utf-8'))
                        
            except Exception as e:
                print(f"接收循环错误: {e}")
                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None
    
    def get_next_command(self) -> Optional[Dict[str, Any]]:
        """获取下一个命令"""
        with self.lock:
            if self.command_queue:
                return self.command_queue.pop(0)
        return None
    
    def send_response(self, response: Dict[str, Any]):
        """发送响应给客户端"""
        if self.client_socket:
            try:
                self.client_socket.send(json.dumps(response).encode('utf-8'))
            except Exception as e:
                print(f"发送响应失败: {e}")
    
    def stop(self):
        """停止服务器"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()


class RobotController:
    """机械臂控制器，处理具体的控制指令"""
    
    def __init__(self, env):
        self.env = env
        self.pose = torch.tensor([0.463,0.0,0.385,0.0086, 0.9216, 0.0205, 0.3875], device=self.env.device).repeat(self.env.num_envs, 1)
        self.gripper_command = False
        self.tcp_position = [0.463,0.0,0.385]
        self.tcp_euler = [3.096287488937378, 5.488191604614258, 0.06342335790395737]
        
    def get_end_effector_position(self) -> Dict[str, float]:
        """获取末端执行器位置"""
        # 这里应该从环境中获取实际的位置信息
        # 目前使用模拟数据
        ee_frame_sensor = self.env.unwrapped.scene["ee_frame"]
        
        self.tcp_position = ee_frame_sensor.data.target_pos_w[0, 0, :].clone().cpu().numpy().tolist()
        #omni.log.warn(f"pos:{tcp_position}")
        tcp_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
        
        tcp_euler_tensor = euler_xyz_from_quat(tcp_orientation)
        
        self.tcp_euler = [tensor.cpu().item() for tensor in tcp_euler_tensor]

        return {
            "x": self.tcp_position[0],
            "y": self.tcp_position[1], 
            "z": self.tcp_position[2],
            "rx": self.tcp_euler[0],
            "ry": self.tcp_euler[1],
            "rz": self.tcp_euler[2]
        }

    def get_obj_position(self) -> Dict[str, float]:
        """获取物体位置"""
        # 这里应该从环境中获取实际的位置信息
        # 目前使用模拟数据
        object_data = self.env.unwrapped.scene["object"].data
        #omni.log.warn(f"obj:{object_data.root_pos_w}")
        object_position = object_data.root_pos_w[0, :].clone().cpu().numpy().tolist()
        
        #omni.log.warn(f"pos:{object_position}")


        return {
            "x": object_position[0],
            "y": object_position[1], 
            "z": object_position[2]
        }
    
    def move_end_effector(self, pose: Dict[str, float]) -> bool:
        """移动末端执行器"""
        try:
            # 更新位置
            target = np.array([
                pose.get("x", self.tcp_position[0]),
                pose.get("y", self.tcp_position[1]),
                pose.get("z", self.tcp_position[2]),
                pose.get("rx", self.tcp_euler[0]),
                pose.get("ry", self.tcp_euler[1]),
                pose.get("rz", self.tcp_euler[2])
            ])
            
            pose_quat = euler_to_quat(target[3:])
            #omni.log.warn(f"quat{pose_quat}")
            pose_target = target[0:3]
            pose_target = np.concatenate([target[0:3], pose_quat])

            #omni.log.warn(f"final{pose_target}")

            # 创建动作向量
            self.pose = torch.tensor(pose_target, device=self.env.device).repeat(self.env.num_envs, 1)

            
            return True
        except Exception as e:
            print(f"移动末端执行器失败: {e}")
            return False
    
    def control_gripper(self, action: str) -> bool:
        """控制夹爪"""

        try:
            if action == "open":
                self.gripper_command = False
                
            elif action == "close":
                self.gripper_command = True
                
            else:
                return False
            
            
            return True
        except Exception as e:
            print(f"控制夹爪失败: {e}")
            return False
    
    def reset_environment(self) -> bool:
        """重置环境"""
        try:
            self.env.reset()
            self.current_position = np.zeros(6)
            self.gripper_command = False
            return True
        except Exception as e:
            print(f"重置环境失败: {e}")
            return False


def pre_process_actions(pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros(pose.shape[0], 1, device=pose.device)
        gripper_vel[:] = -1.0 if gripper_command else 1.0
        # compute actions
        return torch.concat([pose, gripper_vel], dim=1)


def main():
    """Running socket teleoperation receiver with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. "
            "The gripper command will be ignored."
        )

    # 创建socket接收器
    receiver = SocketCommandReceiver(args_cli.socket_host, args_cli.socket_port)
    receiver.start_server()
    
    # 创建机械臂控制器
    robot_controller = RobotController(env)

    # reset environment
    env.reset()
    print("环境已重置，等待控制指令...")

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # 检查是否有新的控制指令
            command = receiver.get_next_command()
            
            if command:
                action = command.get("action")
                
                if action == "get_position":
                    # 获取位置
                    position = robot_controller.get_end_effector_position()
                    #omni.log.warn(f"position:{position}")
                    response = {
                        "success": True,
                        "position": position
                    }
                    receiver.send_response(response)
                    
                elif action == "move_end_effector":
                    # 移动末端执行器
                    pose = command.get("pose", {})
                    success = robot_controller.move_end_effector(pose)
                    response = {
                        "success": success,
                        "error": None if success else "移动失败"
                    }
                    receiver.send_response(response)
                    
                elif action == "get_obj_position":
                    # 获取物体位置
                    obj_pose = robot_controller.get_obj_position()
                    response = {
                        "success": True,
                        "object position": obj_pose
                    }
                    receiver.send_response(response)
                    
                elif action == "control_gripper":
                    # 控制夹爪
                    gripper_action = command.get("gripper_action")
                    success = robot_controller.control_gripper(gripper_action)
                    response = {
                        "success": success,
                        "error": None if success else "夹爪控制失败"
                    }
                    receiver.send_response(response)
                    
                elif action == "reset_environment":
                    # 重置环境
                    success = robot_controller.reset_environment()
                    response = {
                        "success": success,
                        "error": None if success else "环境重置失败"
                    }
                    receiver.send_response(response)
                    
                else:
                    # 未知指令
                    response = {
                        "success": False,
                        "error": f"未知指令: {action}"
                    }
                    receiver.send_response(response)
            action = pre_process_actions(robot_controller.pose, robot_controller.gripper_command)
            
            env.step(action)

    # 停止socket服务器
    receiver.stop()
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 