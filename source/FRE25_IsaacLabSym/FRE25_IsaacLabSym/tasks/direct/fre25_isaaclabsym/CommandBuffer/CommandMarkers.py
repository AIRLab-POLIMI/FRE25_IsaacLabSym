import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# configuration
##

COMMAND_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/Commands",
    markers={
        "arrow_x": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.1, 0.1, 0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 0.5)),
        ),
    },
)
