"""Package containing task implementations for various robotic environments."""

import os
import toml

# 直接导入所有子包来注册环境
from .Robot_arm import *
from .Dextrous_hand import *

# Conveniences to other module directories via relative paths
LAB_TASKS_EXT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)
"""Path to the extension source directory."""

LAB_TASKS_METADATA = toml.load(
    os.path.join(LAB_TASKS_EXT_DIR, "config", "extension.toml")
)
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = LAB_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##
