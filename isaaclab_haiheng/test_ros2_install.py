#!/usr/bin/env python3
"""
测试ROS2依赖是否正确安装
"""


def test_ros2_imports():
    """测试ROS2相关包的导入"""
    try:
        print("正在测试ROS2依赖...")
        
        # 测试核心ROS2包
        import rclpy
        print("✓ rclpy 导入成功")
        
        from rclpy.node import Node
        print("✓ rclpy.node 导入成功")
        
        # 测试消息包
        from geometry_msgs.msg import PoseStamped
        print("✓ geometry_msgs.msg 导入成功")
        
        from std_msgs.msg import Bool
        print("✓ std_msgs.msg 导入成功")
        
        # 测试ROS2初始化
        rclpy.init()
        print("✓ ROS2 初始化成功")
        
        # 创建测试节点
        node = Node('test_node')
        print("✓ 节点创建成功")
        
        # 测试消息创建
        _pose_msg = PoseStamped()
        _bool_msg = Bool()
        print("✓ 消息对象创建成功")
        
        # 清理
        node.destroy_node()
        rclpy.shutdown()
        print("✓ ROS2 清理成功")
        
        print("\n🎉 所有ROS2依赖测试通过！")
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False


if __name__ == "__main__":
    test_ros2_imports() 