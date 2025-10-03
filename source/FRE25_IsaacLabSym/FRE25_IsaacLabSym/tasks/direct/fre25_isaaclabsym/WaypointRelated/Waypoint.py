import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

##
# configuration
##

WAYPOINT_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/Waypoints",
    markers={
        "marker0": sim_utils.SphereCfg(  # Future target (red)
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), opacity=0.5
            ),
        ),
        "marker1": sim_utils.SphereCfg(  # Current targets (green)
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "marker2": sim_utils.SphereCfg(  # Past targets (blue, semi-transparent)
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),
                opacity=0.2,  # set alpha via the opacity field (PreviewSurface expects RGB + separate opacity)
            ),
        ),
    },
)
