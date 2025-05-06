import os
import isaaclab.sim as sim_utils

# Get absolute path to workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# USD path with proper resolution for cross-platform compatibility
USD_PATH = os.path.join(WORKSPACE_ROOT, "Assets", "Plant.usd")

PLANT_CFG = sim_utils.UsdFileCfg(usd_path=USD_PATH, scale=(0.03, 0.03, 0.03))
