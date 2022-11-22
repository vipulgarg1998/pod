import signal
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

# Zed Library
import pyzed.sl as sl

# OpenCV Library
import cv2 as cv

class ZedPublisher(Node):

    def __init__(self):
        super().__init__('zed_publisher')
        # Create Image Publisher
        self.image_pub = self.create_publisher(Image, 'camera/image', 10)
        self.depth_pub = self.create_publisher(Image, 'camera/depth/image', 10)
        fps = 30 
        self.timer = self.create_timer(1/fps, self.publish)

        self.subscription = self.create_subscription(
            Image,
            'depth/image',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Create Zed Camera
        self.zed = sl.Camera()
        self.cv_bridge = CvBridge()

        # Create a InitParameters object and set configuration parameters
        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        self.init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
        # Create a Runtime object and set configuration parameters
        self.runtime_parameters = sl.RuntimeParameters()

        # Zed Camera Details
        self.fx = 0
        self.fy = 0
        self.cx = 0
        self.cy = 0
        self.width = 0
        self.height = 0

        # Close camera on exit
        signal.signal(signal.SIGINT, self.exit)

    def listener_callback(self, msg):
        return
        # cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        # cv.imshow("depth", cv_image)
        # cv.waitKey(1)

    def open_camera(self):
        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # Print Camera Intrinsincs
        self.retrieve_camera_params(debug = True)
    
        # Create zed images
        self.image = sl.Mat(self.width, self.height, sl.MAT_TYPE.U8_C4)
        self.depth = sl.Mat(self.width, self.height, sl.MAT_TYPE.F32_C1)
        self.point_cloud = sl.Mat()

    def retrieve_camera_params(self, debug = True):
        left_cam_params = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam

        self.fx = left_cam_params.fx
        self.fy = left_cam_params.fy
        self.cx = left_cam_params.cx
        self.cy = left_cam_params.cy
        self.width = self.zed.get_camera_information().camera_resolution.width
        self.height = self.zed.get_camera_information().camera_resolution.height

        if(debug):
            print("Fx:", left_cam_params.fx)
            print("Fy:", left_cam_params.fy)
            print("Cx:", left_cam_params.cx)
            print("Cy:", left_cam_params.cy)


    def publish(self):
        if self.zed.is_opened():
            # Grab an image, a RuntimeParameters object must be given to grab()
            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # A new image is available if grab() returns SUCCESS
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)

                cv_image = self.image.get_data()
                cv_image = cv.cvtColor(cv_image, cv.COLOR_RGBA2BGR)

                header=Header()
                header.frame_id = "image"
                header.stamp = self.get_clock().now().to_msg()
                self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, encoding = "rgb8", header=header))



                cv_depth = self.depth.get_data()

                header.frame_id = "depth"
                header.stamp = self.get_clock().now().to_msg()

                # sensor_depth_img = Image()
                # sensor_depth_img.header = header
                # sensor_depth_img.height = self.depth.get_height()
                # sensor_depth_img.width = self.depth.get_width()
                # sensor_depth_img.is_bigendian = 1
                # sensor_depth_img.step = self.depth.get_step_bytes()
                # sensor_depth_img.encoding = "32FC1"
                # sensor_depth_img.data = self.depth.get_data().astype(np.uint8).flatten().tolist()
                # self.depth_pub.publish(sensor_depth_img)
                # header.stamp = self.get_clock().now().to_msg()
                self.depth_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_depth, encoding = "32FC1", header=header))
                # cv.imshow("Image", cv_depth)
                # cv.waitKey(1)

    def exit(self):
        self.zed.close()

def main(args=None):
    
    rclpy.init(args=args)

    zed_publisher = ZedPublisher()
    zed_publisher.open_camera()
    zed_publisher.publish()

    rclpy.spin(zed_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    zed_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()