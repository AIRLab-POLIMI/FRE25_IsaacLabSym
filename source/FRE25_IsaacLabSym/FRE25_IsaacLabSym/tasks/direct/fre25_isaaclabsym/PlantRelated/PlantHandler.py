from isaaclab.assets import (
    Articulation,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)

import isaacsim.core.utils.prims as prim_utils
from ..PathHandler import PathHandler

from .Plant import PLANT_CFG
from ..RayMarcher.RayMarcher import RayMarcher
import torch


class PlantHandler:

    def __init__(self, nPlants: int = 100, envsOrigins: torch.Tensor = None, plantRadius: float = 1):
        assert nPlants > 0, "Number of plants must be greater than 0, got {}".format(nPlants)
        self.nPlants = nPlants

        assert envsOrigins is not None, "envsOrigins must be provided"
        self.envsOrigins = envsOrigins

        assert plantRadius > 0, "Plant radius must be greater than 0, got {}".format(plantRadius)
        self.plantRadius = plantRadius

        self.raymarcher = RayMarcher(
            envsOrigins=self.envsOrigins,
            device=self.envsOrigins.device,
            raysPerRobot=180,  # Number of rays per robot
            maxDistance=10.0,  # Maximum distance for raymarching
            tol=0.01,  # Tolerance for raymarching
            maxSteps=100  # Maximum steps for raymarching
        )
        pass

    def spawnPlants(self):
        # Spawn the plants
        # Ensure the Plants prim path exists
        prim_utils.create_prim("/World/envs/env_0/Plants")

        plantsCFG = RigidObjectCollectionCfg(
            rigid_objects={
                f"Plant_{i}": RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Plants/Plant_{i}",
                    spawn=PLANT_CFG,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0, 0, 0),
                        rot=(0.70711, 0.70711, 0, 0),

                    ),
                ) for i in range(self.nPlants)
            }
        )
        self.plants: RigidObjectCollection = RigidObjectCollection(plantsCFG)

    def randomizePlantsPositions(self, env_ids: torch.Tensor, pathHandler: PathHandler):
        # Randomize plant positions
        envOrigins: torch.Tensor = self.envsOrigins[env_ids]
        envOrigins = envOrigins.unsqueeze(1).repeat(
            1, self.nPlants, 1
        )

        objStates = self.plants.data.object_state_w[env_ids]

        pathPosition = pathHandler.samplePoints(env_ids, self.nPlants)
        objStates[:, :, :2] = envOrigins[:, :, :2] + pathPosition

        self.plants.write_object_state_to_sim(objStates, env_ids)

    def computeDistancesToPlants(self, robot_pos_xy: torch.Tensor) -> torch.Tensor:
        """
        Compute distances from the robot to all plants.
        :param robot_pos_xy: Robot position in world coordinates (nEnvs, 2)
        :return: A tensor of distances from the robot to each plant in each environment (nEnvs, nPlants)
        """
        assert robot_pos_xy.dim() == 2, "robot_pos_xy must be a 2D tensor, got {}D".format(robot_pos_xy.dim())
        _, distances, _ = self.raymarcher.sense(
            robot_pos_xy=robot_pos_xy,
            plantsPositions=self.plants.data.object_state_w[:, :, :2],
            plantRadius=self.plantRadius
        )
        self.distances = distances
        return distances

    def detectPlantCollision(self) -> torch.Tensor:
        """
        Detect if the robot is colliding with any plant.
        :param robot_pos_xy: Robot position in world coordinates (nEnvs, 2)
        :return: A boolean tensor indicating if the robot is colliding with any plant in each environment (nEnvs,)
        """
        # Check if any distance is less than the plant radius
        collisions = (self.distances < self.plantRadius).any(dim=1)
        return collisions
