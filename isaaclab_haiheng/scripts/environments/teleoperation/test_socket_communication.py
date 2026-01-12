#!/usr/bin/env python3
"""
测试socket通信功能的脚本。

这个脚本用于测试controller_mcp.py和teleop_se3_receiver.py之间的通信是否正常工作。
"""

import socket
import json
import time
from typing import Dict, Any


class SocketTestClient:
    """Socket测试客户端"""
    
    def __init__(self, host: str = "localhost", port: int = 12345):
        self.host = host
        self.port = port
        self.socket = None
        
    def connect(self) -> bool:
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            print(f"成功连接到 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """发送命令并接收响应"""
        if not self.socket:
            return {"success": False, "error": "未连接到服务器"}
            
        try:
            # 发送命令
            command_str = json.dumps(command)
            self.socket.send(command_str.encode('utf-8'))
            print(f"发送命令: {command_str}")
            
            # 等待一小段时间让服务器处理命令
            time.sleep(0.1)
            
            # 接收响应
            response = self.socket.recv(4096).decode('utf-8')
            response_data = json.loads(response)
            print(f"收到响应: {response}")
            return response_data
        except Exception as e:
            print(f"通信错误: {e}")
            return {"success": False, "error": str(e)}
    
    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
            self.socket = None


def test_communication():
    """测试通信功能"""
    print("=== Socket通信测试 ===")
    
    # 创建测试客户端
    client = SocketTestClient()
    
    # 尝试连接
    if not client.connect():
        print("无法连接到服务器，请确保teleop_se3_receiver.py正在运行")
        return
    
    try:
        # 测试1: 获取位置
        time.sleep(1)
        print("\n--- 测试1: 获取机械臂末端位置 ---")
        command = {
            "action": "get_position",
            "timestamp": time.time()
        }
        response = client.send_command(command)
        if response.get("success"):
            print("✓ 位置查询成功")
        else:
            print(f"✗ 位置查询失败: {response.get('error')}")
        
        time.sleep(1)

        print("\n--- 测试1.5: 获取物体位置 ---")
        command = {
            "action": "get_obj_position",
            "timestamp": time.time()
        }
        response = client.send_command(command)
        if response.get("success"):
            print("✓ 位置查询成功")
        else:
            print(f"✗ 位置查询失败: {response.get('error')}")
        
        time.sleep(1)
        
        # 测试2: 移动机械臂
        print("\n--- 测试2: 移动机械臂末端 ---")
        command = {
            "action": "move_end_effector",
            "pose": {
                "x": 0.563,
                "y": 0.0,
                "z": 0.385,
                "rx": 3.096287488937378, 
                "ry": 5.488191604614258,
                "rz": 0.06342335790395737
            },
            "timestamp": time.time()
        }
        response = client.send_command(command)
        if response.get("success"):
            print("✓ 移动指令执行成功")
        else:
            print(f"✗ 移动指令执行失败: {response.get('error')}")
        
        time.sleep(1)
        
        # 测试3: 控制夹爪
        print("\n--- 测试3: 控制夹爪 ---")
        command = {
            "action": "control_gripper",
            "gripper_action": "open",
            "timestamp": time.time()
        }
        response = client.send_command(command)
        print(response)
        if response.get("success"):
            print("✓ 夹爪打开成功")
        else:
            print(f"✗ 夹爪打开失败: {response.get('error')}")
        
        time.sleep(1)
        
        command = {
            "action": "control_gripper",
            "gripper_action": "close",
            "timestamp": time.time()
        }
        response = client.send_command(command)
        if response.get("success"):
            print("✓ 夹爪闭合成功")
        else:
            print(f"✗ 夹爪闭合失败: {response.get('error')}")
        
        time.sleep(1)
        
        # 测试4: 重置环境
        print("\n--- 测试4: 重置环境 ---")
        command = {
            "action": "reset_environment",
            "timestamp": time.time()
        }
        response = client.send_command(command)
        if response.get("success"):
            print("✓ 环境重置成功")
        else:
            print(f"✗ 环境重置失败: {response.get('error')}")
        
        # 测试5: 无效命令
        print("\n--- 测试5: 无效命令处理 ---")
        command = {
            "action": "invalid_action",
            "timestamp": time.time()
        }
        response = client.send_command(command)
        if not response.get("success"):
            print("✓ 无效命令被正确处理")
        else:
            print("✗ 无效命令处理异常")
        
        print("\n=== 测试完成 ===")
        
    finally:
        client.close()


def test_connection_only():
    """仅测试连接"""
    print("=== 连接测试 ===")
    
    client = SocketTestClient()
    if client.connect():
        print("✓ 连接成功")
        client.close()
    else:
        print("✗ 连接失败")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--connection-only":
        test_connection_only()
    else:
        test_communication() 