import numpy as np
from record3d import Record3DStream
import cv2
from threading import Event
from stretch.perception import create_semantic_sensor
import matplotlib.pyplot as plt
from stretch.agent.robot_agent import RobotAgent
import time
from stretch.utils.dummy_stretch_client import DummyStretchClient
import torch
from stretch.core import get_parameters
from stretch.mapping.voxel import SparseVoxelMap
import numpy as np
from quaternion import as_rotation_matrix, quaternion
from demo_record3d_dataset import LiveR3DDataset

def as_pose_matrix(pose: list[float]) -> np.ndarray:
    """Converts a list of pose parameters to a pose matrix.

    Args:
        pose: The list of pose parameters, (qx, qy, qz, qw, px, py, pz)

    Returns:
        A (4, 4) pose matrix
    """

    mat = np.eye(4, dtype=np.float64)
    qx, qy, qz, qw, px, py, pz = pose
    mat[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
    mat[:3, 3] = [px, py, pz]
    return mat

class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        self.voxel_map = SparseVoxelMap(resolution=0.04)

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])
    
    def preprocess_pose(self, pose: torch.Tensor) -> torch.Tensor:
        # Step 1: Flip Y and Z axes
        flip_yz = torch.tensor([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1],
        ], dtype=torch.float32, device=pose.device)

        pose = pose @ flip_yz  # First flip Y and Z

        # Step 2: Swap Y and Z
        swap_yz = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=pose.device)

        pose = swap_yz @ pose  # Then rotate
        return pose

    def start_processing_stream(self, run_seconds=20, config_path='/home/xin3/Desktop/stretch_ai_xin/src/stretch/config/app/vlm_planning/multi_crop_vlm_planner.yaml'):
        semantic_sensor = create_semantic_sensor(config_path=config_path)
        # vlm_parameters = get_parameters(config_path)
        # loaded_voxel_map = None
        # robot = DummyStretchClient()
        # agent = RobotAgent(
        #     robot,
        #     vlm_parameters,
        #     voxel_map=loaded_voxel_map,
        #     semantic_sensor=semantic_sensor,
        # )
        # voxel_map = agent.get_voxel_map()

        plt.ion()  # Turn on interactive mode
        fig, axes = None, None
        im1, im2, im3, im4 = None, None, None, None
        start_time = time.time()
        i = 0

        while True:
            if time.time() - start_time > run_seconds:
                print(f"Stream stopped after {run_seconds} seconds.")
                break
            self.event.wait()  # Wait for new frame to arrive

            # Copy the newly arrived RGBD frame
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            confidence = self.session.get_confidence_frame()
            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
            camera_pose = self.session.get_camera_pose()  # Quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])

            print(intrinsic_mat)

            # You can now e.g. create point cloud by projecting the depth map using the intrinsic matrix.

            # Postprocess it
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # # Show the RGBD Stream
            # cv2.imshow('RGB', rgb)
            # cv2.imshow('Depth', depth)
            if confidence.shape[0] > 0 and confidence.shape[1] > 0:
                # cv2.imshow('Confidence', confidence * 100)
                confidence *= 100

            mask, instance, info = semantic_sensor.predict_segmentation(
                    rgb=rgb, depth=depth, base_pose=None
                )
            # After getting rgb and depth, before converting to tensors
            if rgb.shape[:2] != depth.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Example: convert to torch tensors (adjust dtype/device as needed)
            # Extract values from camera_pose object
            pose_list = [
                camera_pose.qx,
                camera_pose.qy,
                camera_pose.qz,
                camera_pose.qw,
                camera_pose.tx,
                camera_pose.ty,
                camera_pose.tz,
            ]
            camera_pose_tensor = torch.tensor(as_pose_matrix(pose_list), dtype=torch.float32)
            camera_pose_tensor = self.preprocess_pose(camera_pose_tensor)
            rgb_tensor = torch.tensor(rgb, dtype=torch.float32)
            depth_tensor = torch.tensor(depth, dtype=torch.float32)
            camera_K_tensor = torch.tensor(intrinsic_mat, dtype=torch.float32)
            instance_image_tensor = torch.tensor(mask, dtype=torch.int64)  # or appropriate dtype
            instance_classes = info["instance_classes"]
            instance_scores = info["instance_scores"]
            
            self.voxel_map.add(
                camera_pose=camera_pose_tensor,
                rgb=rgb_tensor,
                feats=None,
                depth=depth_tensor,
                base_pose=None,
                instance_image=instance_image_tensor,
                instance_classes=instance_classes,
                instance_scores=instance_scores,
                camera_K=camera_K_tensor,
                pose_correction=None,
                # Add other arguments as needed
            )

            if fig is None:
                fig, axes = plt.subplots(1, 4, figsize=(24, 4))
                im1 = axes[0].imshow(rgb)
                axes[0].set_title(f"Frame {i} RGB")
                axes[0].axis("off")
                im2 = axes[1].imshow(mask)
                axes[1].set_title(f"Frame {i} Instance Segmentation")
                axes[1].axis("off")
                im3 = axes[2].imshow(confidence, cmap='gray', vmin=0, vmax=1)
                axes[2].set_title(f"Frame {i} Confidence")
                axes[2].axis("off")
                im4 = axes[3].imshow(depth, cmap='gray')
                axes[3].set_title(f"Frame {i} Depth")
                axes[3].axis("off")
            else:
                im1.set_data(rgb)
                axes[0].set_title(f"Frame {i} RGB")
                im2.set_data(mask)
                axes[1].set_title(f"Frame {i} Instance Segmentation")
                im3.set_data(confidence)
                axes[2].set_title(f"Frame {i} Confidence")
                im4.set_data(depth)
                axes[3].set_title(f"Frame {i} Depth")
            plt.pause(0.05)
            fig.canvas.flush_events()

            cv2.waitKey(1)
            i+=1

            self.event.clear()

if __name__ == '__main__':
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream(run_seconds=20)
    app.voxel_map.show()
