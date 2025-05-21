import numpy as np
import torch
import cv2
from torch.utils.data.dataset import Dataset
from record3d import Record3DStream
from threading import Event
from typing import NamedTuple, Optional, Tuple
from quaternion import as_rotation_matrix, quaternion
import torchvision.transforms.functional as V

class PosedRGBDItem(NamedTuple):
    image: torch.Tensor
    depth: torch.Tensor
    mask: torch.Tensor
    intrinsics: torch.Tensor
    pose: torch.Tensor

    def check(self) -> None:
        assert self.image.dim() == 3
        assert self.image.dtype == torch.float32
        assert self.depth.dim() == 3
        assert self.depth.shape[0] == 1
        assert self.depth.dtype == torch.float32
        assert self.depth.shape[1:] == self.image.shape[1:]
        assert self.mask.shape[1:] == self.image.shape[1:]
        assert self.intrinsics.shape == (3, 3)
        assert self.pose.shape == (4, 4)

def as_pose_matrix(pose: list[float]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    qx, qy, qz, qw, px, py, pz = pose
    mat[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
    mat[:3, 3] = [px, py, pz]
    return mat

class LiveR3DDataset(Dataset[PosedRGBDItem]):
    def __init__(
        self,
        device_idx: int = 0,
        use_depth_shape: bool = True,
        shape: Optional[Tuple[int, int]] = None,
        x1: float = None,
        x2: float = None,
        y1: float = None,
        y2: float = None,
        z_offset: float = None,
        buffer_size: int = 10
    ) -> None:
        """A dataset for live RGB-D streaming from Record3D.

        Args:
            device_idx: Index of the Record3D device to connect to.
            use_depth_shape: If True, resize RGB to depth shape; otherwise, resize depth to RGB shape.
            shape: Optional custom shape (height, width) to resize both RGB and depth.
            x1, x2, y1, y2, z_offset: Parameters for pose transformation (optional).
            buffer_size: Number of frames to keep in the buffer.
        """
        self.use_depth_shape = use_depth_shape
        self.shape = shape
        self.buffer_size = buffer_size
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

        # Initialize buffer for frames
        self.frame_buffer = []
        self.frame_count = 0

        # Pose transformation parameters
        self.x1, self.x2, self.y1, self.y2, self.z_offset = x1, x2, y1, y2, z_offset

        # Connect to Record3D device
        self.connect_to_device(device_idx)

    def connect_to_device(self, dev_idx: int) -> None:
        devs = Record3DStream.get_connected_devices()
        if len(devs) <= dev_idx:
            raise RuntimeError(f"Cannot connect to device #{dev_idx}, try different index.")
        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)

    def on_new_frame(self):
        self.event.set()  # Signal new frame arrival

    def on_stream_stopped(self):
        print("Stream stopped")

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx, 0, coeffs.tx],
                        [0, coeffs.fy, coeffs.ty],
                        [0, 0, 1]], dtype=np.float64)

    def process_frame(self) -> Optional[PosedRGBDItem]:
        """Process a single frame from the stream."""
        if not self.event.is_set():
            return None

        # Get frame data
        depth = self.session.get_depth_frame()
        rgb = self.session.get_rgb_frame()
        confidence = self.session.get_confidence_frame()
        intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
        camera_pose = self.session.get_camera_pose()

        # Postprocess
        if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
            depth = cv2.flip(depth, 1)
            rgb = cv2.flip(rgb, 1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Resize RGB and depth to match
        rgb_h, rgb_w = rgb.shape[:2]
        depth_h, depth_w = depth.shape[:2]

        if self.shape is not None:
            arr_h, arr_w = self.shape
            rgb = cv2.resize(rgb, (arr_w, arr_h), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (arr_w, arr_h), interpolation=cv2.INTER_NEAREST)
            if confidence.shape[0] > 0 and confidence.shape[1] > 0:
                confidence = cv2.resize(confidence, (arr_w, arr_h), interpolation=cv2.INTER_NEAREST)
        elif self.use_depth_shape:
            if (rgb_h, rgb_w) != (depth_h, depth_w):
                rgb = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
        else:
            if (rgb_h, rgb_w) != (depth_h, depth_w):
                depth = cv2.resize(depth, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
                if confidence.shape[0] > 0 and confidence.shape[1] > 0:
                    confidence = cv2.resize(confidence, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)

        # Adjust intrinsics
        if self.shape is not None:
            scale_x = arr_w / rgb_w
            scale_y = arr_h / rgb_h
            intrinsic_mat[0, :] *= scale_x
            intrinsic_mat[1, :] *= scale_y
        elif self.use_depth_shape and (rgb_h, rgb_w) != (depth_h, depth_w):
            scale_x = depth_w / rgb_w
            scale_y = depth_h / rgb_h
            intrinsic_mat[0, :] *= scale_x
            intrinsic_mat[1, :] *= scale_y
        elif not self.use_depth_shape and (rgb_h, rgb_w) != (depth_h, depth_w):
            scale_x = rgb_w / depth_w
            scale_y = rgb_h / depth_h
            intrinsic_mat[0, :] *= scale_x
            intrinsic_mat[1, :] *= scale_y

        # Process confidence to create mask
        if confidence.shape[0] > 0 and confidence.shape[1] > 0:
            confidence = confidence * 100
            mask = (confidence != 200).astype(np.uint8)  # Mimic R3DDataset: conf != 2
        else:
            mask = np.ones_like(depth, dtype=np.uint8)
            mask[depth < 3] = 0  # Mimic R3DDataset fallback

        # Handle NaN in depth
        depth_is_nan = np.isnan(depth)
        depth[depth_is_nan] = -1.0
        mask[depth_is_nan] = 0

        # Convert pose
        pose_list = [
            camera_pose.qx,
            camera_pose.qy,
            camera_pose.qz,
            camera_pose.qw,
            camera_pose.tx,
            camera_pose.ty,
            camera_pose.tz,
        ]
        pose = as_pose_matrix(pose_list)

        # Apply pose transformations (same as R3DDataset)
        affine_matrix = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ])
        pose = pose @ affine_matrix
        affine_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
        pose = affine_matrix @ pose

        if all(v is not None for v in [self.x1, self.x2, self.y1, self.y2, self.z_offset]):
            x_offset, y_offset = self.x1, self.y1
            theta_offset = np.arctan2((self.y2 - self.y1), (self.x2 - self.x1))
            n2r_matrix = np.array([
                [np.cos(theta_offset), np.sin(theta_offset), 0, 0],
                [-np.sin(theta_offset), np.cos(theta_offset), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]) @ np.array([
                [1, 0, 0, -x_offset],
                [0, 1, 0, -y_offset],
                [0, 0, 1, -self.z_offset],
                [0, 0, 0, 1]
            ])
            pose = n2r_matrix @ pose

        # Convert to tensors
        img = torch.from_numpy(rgb).permute(2, 0, 1)
        img = V.convert_image_dtype(img, torch.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        intr = torch.from_numpy(intrinsic_mat)
        pose = torch.from_numpy(pose)

        item = PosedRGBDItem(image=img, depth=depth, mask=mask, intrinsics=intr, pose=pose)
        item.check()

        # Add to buffer
        self.frame_buffer.append(item)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        self.frame_count += 1

        self.event.clear()
        return item

    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, index: int) -> PosedRGBDItem:
        # Wait for a new frame if buffer is empty or index is beyond current frames
        while len(self.frame_buffer) == 0 or index >= self.frame_count:
            item = self.process_frame()
            if item is not None:
                return item
        return self.frame_buffer[min(index, len(self.frame_buffer) - 1)]