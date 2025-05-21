import numpy as np
from record3d import Record3DStream
import cv2
from threading import Event
import open3d as o3d  # You'll need to install this: pip install open3d

class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        self.vis = None  # Open3D visualizer
        self.point_cloud = None  # Open3D point cloud object
        self.is_vis_initialized = False

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

    def create_point_cloud(self, rgb, depth, intrinsic_mat):
        """
        Create a colored point cloud from RGB and depth data.
        """
        # Create coordinate grid
        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Flatten all arrays
        u = u.flatten()
        v = v.flatten()
        z = depth.flatten()
        
        # Remove invalid depth values (0 or very large values often indicate invalid measurements)
        valid_depth = (z > 0) & (z < 10000)  # Adjust the upper threshold as needed
        u = u[valid_depth]
        v = v[valid_depth]
        z = z[valid_depth]
        
        # Calculate 3D coordinates using the intrinsic matrix
        fx = intrinsic_mat[0, 0]
        fy = intrinsic_mat[1, 1]
        cx = intrinsic_mat[0, 2]
        cy = intrinsic_mat[1, 2]
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Combine coordinates into points array
        points = np.vstack((x, y, z)).T
        
        # Get corresponding colors
        colors = rgb[v, u] / 255.0  # Normalize color values to [0, 1]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def initialize_visualizer(self):
        """
        Initialize the Open3D visualizer.
        """
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Point Cloud Viewer", width=1024, height=768)
        
        # Add coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(coord_frame)
        
        # Create an empty point cloud and add it to the visualizer
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)
        
        # Set some reasonable default viewpoint
        view_control = self.vis.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])
        
        self.is_vis_initialized = True

    def update_point_cloud(self, new_pcd):
        """
        Update the point cloud in the visualizer.
        """
        self.point_cloud.points = new_pcd.points
        self.point_cloud.colors = new_pcd.colors
        self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()

    def start_processing_stream(self):
        if not self.is_vis_initialized:
            self.initialize_visualizer()
            
        while True:
            self.event.wait()  # Wait for new frame to arrive
            
            # Copy the newly arrived RGBD frame
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            confidence = self.session.get_confidence_frame()
            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
            camera_pose = self.session.get_camera_pose()  # Quaternion + world position
            
            # Postprocess frames
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)
            
            # Keep RGB in RGB format for point cloud but convert to BGR for OpenCV display
            rgb_for_display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Create point cloud
            pcd = self.create_point_cloud(rgb, depth, intrinsic_mat)
            
            # Update visualizer with new point cloud
            self.update_point_cloud(pcd)
            
            # Show the RGBD Stream
            cv2.imshow('RGB', rgb_for_display)
            cv2.imshow('Depth', depth)
            if confidence.shape[0] > 0 and confidence.shape[1] > 0:
                cv2.imshow('Confidence', confidence * 100)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
                
            self.event.clear()
        
        # Clean up
        cv2.destroyAllWindows()
        self.vis.destroy_window()

if __name__ == '__main__':
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()