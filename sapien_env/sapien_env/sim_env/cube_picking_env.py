# Defines table, cube placement, cube randomization, and lighting
import numpy as np
import sapien.core as sapien
import transforms3d.euler

from sapien_env.sim_env.base import BaseSimulationEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.gui_base import GUIBase, YX_TABLE_TOP_CAMERAS

class CubePickingEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, randomness_scale=1, seed=None,
                 use_ray_tracing=True, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, use_ray_tracing=use_ray_tracing, **renderer_kwargs)

        self.randomness_scale = randomness_scale
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)

        self.table = self.create_table(table_height=0.6, table_half_size=[0.35, 0.7, 0.025])
        self.cube = self._create_cube()
        self.cube.set_pose(sapien.Pose([0.1, 0, 0.62]))
        self.original_object_pos = np.zeros(3)

    def _create_cube(self):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.02], color=[1.0, 0.1, 0.1, 1])
        cube = builder.build(name="cube")
        return cube

    def reset_env(self):
        pose = self.generate_random_init_pose(self.randomness_scale)
        self.cube.set_pose(pose)
        self.original_object_pos = pose.p

    def generate_random_init_pose(self, randomness_scale):
        pos = np.array([
            self.np_random.uniform(0.0, 0.15),
            self.np_random.uniform(-0.15, 0.15),
            0.62  # 0.6 + 0.02  # table height + half cube height
        ])
        quat = transforms3d.euler.euler2quat(0, 0, 0)
        return sapien.Pose(pos, quat)

    def get_init_poses(self):
        return np.stack([self.cube.get_pose().to_transformation_matrix()])


def env_test():
    from sapien.utils import Viewer
    env = CubePickingEnv(use_ray_tracing=False)
    env.reset_env()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi / 2)
    viewer.set_fovy(2.0)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


def get_init_pic():
    import cv2, os
    output_dir = f"/home/hz2999/gendp/data/cube"
    os.makedirs(output_dir, exist_ok=True)
    env = CubePickingEnv(use_ray_tracing=False, seed=0)
    env.reset_env()
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer, headless=True)

    for name, params in YX_TABLE_TOP_CAMERAS.items():
        if 'rotation' in params:
            gui.create_camera_from_pos_rot(**params)
        else:
            gui.create_camera(**params)

    for _ in range(50):
        env.simple_step()
        rgbs = gui.render()

    for i, rgb in enumerate(rgbs):
        cv2.imwrite(f'{output_dir}/{i}.png', rgb)


if __name__ == '__main__':
    #env_test()
    get_init_pic()