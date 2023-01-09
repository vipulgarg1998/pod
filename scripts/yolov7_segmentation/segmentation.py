#!/usr/bin/env python3
# ROS Libs
import rospy
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from std_msgs.msg import Header, Empty
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

from pose_mimic.srv import Objects, ObjectsResponse

# Python Libs
import sys
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time
import yaml
import ctypes
 
# Open3D
# import open3d as o3d
# import numpy as np

# Yolo Libs
from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf
import torch
from torchvision import transforms, ops
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

# References
# https://github.com/WongKinYiu/yolov7/tree/pose
# https://viso.ai/computer-vision/coco-dataset/
# https://learnopencv.com/yolov7-pose-vs-mediapipe-in-human-pose-estimation/
# https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/
 
class PoseEstimator:
    def __init__(self, model_filename = '../../weights/yolov7-mask.pt', server_deactivated = False):
        # super().__init__('minimal_publisher')


        tss = ApproximateTimeSynchronizer(
            [Subscriber("zed/zed_node/rgb/image_rect_color", Image), 
            Subscriber("zed/zed_node/depth/depth_registered", Image)], 10, 1)
        tss.registerCallback(self.rgbd_callback)
        # self.image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.image_callback)
        self.image_sub = rospy.Subscriber("/yolo/segmentation/stop", Empty, self.stop_callback)
        self.keypoint_pub = rospy.Publisher('centroids', PointCloud2, queue_size=10)
        self.marker_sub = rospy.Subscriber("visualization_marker", Marker, self.marker_callback)
        self.camera_info_sub = rospy.Subscriber("/zed/zed_node/left/camera_info", CameraInfo, self.camera_info_callback)

        # For OpenCV
        self.cv_bridge = CvBridge()
        self.cv_image = None

        # For Objects
        self.centroids = []
        self.segments_3d = []
        self.vector_p1 = [0.0, 0.0, 0.0]
        self.vector_p2 = [0.0, 0.0, 0.0]

        # For Yolo 
        self.model_filename = model_filename
        self.device = None
        self.weights = None
        self.model = None

        # Segmentation Model
        self.hyp = None
        self.seg_model_filename = model_filename
        self.seg_model_config_filename = "data/hyp.scratch.mask.yaml"
        self.get_objects_srv = '/yolo/segmentation/objects'
        self.seg_weights = None
        self.seg_model = None
        self.model_loaded = False
        self.stop_segmentation = False
        self.seg_in_prog = False
        self.server_deactivated = server_deactivated
        # Service for pose_mimic
        if(not server_deactivated):
            s = rospy.Service(self.get_objects_srv, Objects, self.retrieve_objects)

        # For Camera 
        # self.fx = 1404.6019287109375
        # self.fy = 1404.6019287109375 
        # self.cx = 948.8173217773438
        # self.cy = 557.4688110351562
        self.caliberation_params_init = False
        self.fx = None
        self.fy = None 
        self.cx = None
        self.cy = None
        self.width = None
        self.height = None

        # PointCloud2 Fields
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('intensity', 12, PointField.FLOAT32, 1)]

        self.frame_id = "zed_left_camera_optical_frame"

    def stop_callback(self, msg):
        print("Stop the Segmentation")
        self.stop_segmentation = True
        sys.exit("Request to Stop the Segmentation")


    def retrieve_objects(self, req):
        print("Service Requested")
        if(not self.model_loaded):
            self.load_segmentation_model()
        labels = []
        x = []
        y = []
        z = []
        self.server_deactivated = True
        while(len(self.centroids) == 0):
            time.sleep(0.001)
        # print(self.centroids[0])
        for i in self.centroids[0]:
            # print(i, self.centroids[0][i])
            labels.append(i)
            xyz = self.centroids[0][i]
            x.append(xyz[0])
            y.append(xyz[1])
            z.append(xyz[2])
        self.centroids = []
        self.server_deactivated = False
        objs = ObjectsResponse()
        objs.labels = labels
        objs.x = x
        objs.y = y
        objs.z = z

        if(len(labels) > 0):
            self.unload_segmentation_model()

        return objs
        
    def unload_segmentation_model(self):
        del self.seg_model
        del self.seg_weights
        del self.hyp
        self.model_loaded = False
        torch.cuda.empty_cache() # PyTorch thing
        print("Segmenation Model Unloaded")

    def load_segmentation_model(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(self.seg_model_config_filename) as f:
            self.hyp = yaml.load(f, Loader=yaml.FullLoader)
        self.seg_weigths = torch.load(self.seg_model_filename)
        self.seg_model = self.seg_weigths['model']
        self.seg_model = self.seg_model.half().to(self.device)
        _ = self.seg_model.eval()
        self.model_loaded = True
        print("Segmenation Model Loaded")

    def camera_info_callback(self, camera_info_msg):
        K = camera_info_msg.K
        self.fx = float(K[0])
        self.cx = float(K[2])
        self.fy = float(K[4])
        self.cy = float(K[5])
        self.width = int(camera_info_msg.width)
        self.height = int(camera_info_msg.height)
        self.caliberation_params_init = True

    def marker_callback(self, msg):
        self.vector_p1 = [msg.points[0].x, msg.points[0].y, msg.points[0].z]
        self.vector_p2 = [msg.points[1].x, msg.points[1].y, msg.points[1].z]
        return

    def rgbd_callback(self, image_msg, depth_image_msg):
        print("RGBD Callback")
        if(not self.model_loaded or not self.caliberation_params_init):
            print("Issue in Segmentation")
            return
        if(not self.server_deactivated):
            print("Server Deactivated")
            return

        self.seg_in_prog = True
        print("Starting Segmentation")
        if(self.stop_segmentation):
            # sensor_msg = self.to_sensor_msgs(self.segments_3d)
            # self.keypoint_pub.publish(sensor_msg)
            sensor_msg = self.publish_centroids(self.centroids)
            self.keypoint_pub.publish(sensor_msg)
        else:
            cv_depth_image = self.cv_bridge.imgmsg_to_cv2(depth_image_msg)
            segments, ratio = self.image_callback(image_msg, view = False)
            segments_3d = self.cvrt_2d_to_3d(segments, ratio, cv_depth_image, self.fx, self.fy, self.cx, self.cy, view=False)
            self.segments_3d.append(segments_3d)
            self.centroids.append(self.get_centroids(segments_3d))

        self.seg_in_prog = False

    def publish_centroids(self, centroids_list):
        points_3d = []
        for centroids in centroids_list:
            # print(centroids)
            for i in centroids: 
                # print(centroids[i])
                centroid = list(centroids[i])
                points_3d.append(centroid)
        # print("Printing Points 3D")
        # print(points_3d)
        points_3d = np.array(points_3d)
        header = Header()
        header.frame_id = self.frame_id
        header.stamp = rospy.Time.now()

        if(points_3d.shape[0] == 0): # No object found
            return PointCloud2()

        # Add intensity = 1 to the 4th column
        points_3d = np.hstack((points_3d,np.ones([points_3d.shape[0],1], points_3d.dtype)))

        pc2 = point_cloud2.create_cloud(header, self.fields, points_3d)
        return pc2

    def to_sensor_msgs(self, centroids, segments = True):
        points_3d = []
        for i in centroids:
            # print(centroids[i][0])
            centroid = list(centroids[i])
            if(not segments):
                points_3d.append(centroid)
            else:
                points_3d.extend(centroids[i])
        
        # min_dist = 1000
        # object_selected = None
        # for idx, i in enumerate(centroids):
        #     centroid = list(centroids[i])
        #     dist = self.get_dist(self.vector_p1, self.vector_p2, centroid)
        #     if(dist < min_dist):
        #         object_selected = i
        #         min_dist = dist
        
        # print(points_3d)
        # print(f"Object Selected is {object_selected}")
        points_3d = np.array(points_3d)
        # print(points_3d.shape)
        # print(points_3d)
        header = Header()
        header.frame_id = self.frame_id
        header.stamp = rospy.Time.now()

        if(points_3d.shape[0] == 0): # No object found
            return PointCloud2()

        # Add intensity = 1 to the 4th column
        points_3d = np.hstack((points_3d,np.ones([points_3d.shape[0],1], points_3d.dtype)))

        pc2 = point_cloud2.create_cloud(header, self.fields, points_3d)
        return pc2

    def get_dist(self, vector_p1, vector_p2, point):
        vector_p1 = np.array(vector_p1)
        vector_p2 = np.array(vector_p2)
        point = np.array(point)
        return np.linalg.norm(np.cross(vector_p2-vector_p1, vector_p1-point))/np.linalg.norm(vector_p2-vector_p1)

    def image_callback(self, image_msg, view = True):
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
        # cv_image = self.preprocess_images(cv_image, new_shape=(640, 640))
        tensor_image, ratio = self.convert_image(cv_image)
        segments = self.apply_segmentation(tensor_image, view)
        return segments, ratio

    def convert_image(self, image):
        image, ratio, _ = letterbox(image, stride=64, auto=False, scaleFill = True)
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        image = image.to(self.device)
        image = image.half()
        return image, ratio

    def apply_segmentation(self, image, view = False):
        segments = {}

        output = self.seg_model(image)
        inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']
        bases = torch.cat([bases, sem_output], dim=1)
        nb, _, height, width = image.shape
        names = self.seg_model.names
        pooler_scale = self.seg_model.pooler_scale
        pooler = ROIPooler(output_size=self.hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)
        output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, self.hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)
        pred, pred_masks = output[0], output_mask[0]
        base = bases[0]
        bboxes = Boxes(pred[:, :4])
        original_pred_masks = pred_masks.view(-1, self.hyp['mask_resolution'], self.hyp['mask_resolution'])
        pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
        pred_masks_np = pred_masks.detach().cpu().numpy()
        pred_cls = pred[:, 5].detach().cpu().numpy()
        pred_conf = pred[:, 4].detach().cpu().numpy()
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
        pnimg = nimg.copy()
        for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
            label = names[int(cls)]
            if conf < 0.25 or label == 'person' or label == "bed":
                continue

            segments[label] = one_mask
            if view:            
                color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
                                    
                pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
                pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        if view:
            cv2.imshow('image', pnimg)
            # Press `q` to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.exit()

        return segments

    def cvrt_2d_to_3d(self, segments, ratio, cv_depth_image, fx, fy, cx, cy, view = True):
        segments_3d = {}
        for i in segments: # For each object
            points_3d = []
            mask_points = segments[i]
            coords = np.where(mask_points)
            coords_u = coords[0]
            coords_v = coords[1]
            # print(np.where(mask_points))
            # print(i, mask_points.shape)
            # for u in range(640):
            #     for v in range(640):
            #         if(mask_points[u, v]):
            #             coords.append([u,v])
            for point in zip(list(coords_u), list(coords_v)):
                x_coord, y_coord = point[1], point[0]
                if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                    x_coord_original, y_coord_original = int(x_coord/ratio[0]), int(y_coord/ratio[1])
                    if(x_coord_original >= self.width or y_coord_original >= self.height):
                        continue
                    # print("COordinates ", x_coord, " Y ", y_coord)
                    # print("Original COordinates ", x_coord_original, " Y ", y_coord_original)
                    z = cv_depth_image[int(y_coord_original), int(x_coord_original)]

                    # Check if depth is inconsistent
                    if(not np.isfinite(z)):
                        continue
                    x = (x_coord_original - cx)*z/fx
                    y = (y_coord_original - cy)*z/fy
                    data_point = [x, y, z]
                    points_3d.append(data_point)
            # print(points_3d)
            segments_3d[i] = points_3d
        return segments_3d
    
    def get_centroids(self, segments_3d):
        centroids = {}
        for i in segments_3d:
            if(len(segments_3d[i]) < 1):
                continue
            centroid = np.mean(np.array(segments_3d[i]), axis=0)
            # print(centroid.shape)
            # if(centroid.shape[0] != 3):
            #     continue
            centroids[i] = centroid
            print(i, centroids[i])
            # if(len(segments_3d[i]) > 0):
            #     centroid = np.mean(np.array(segments_3d[i]), axis=0)
            #     # if(not np.isnan(centroid).any()):
            #     centroids[i] = centroid
            #     print(i, centroids[i])
        return centroids

    def exit(self):
        cv2.destroyAllWindows()
        quit()

def main(args=None):
    
    # initializing the subscriber node
    rospy.init_node('pose_estimate', anonymous=True)

    pose_estimator = PoseEstimator(server_deactivated = False)
    pose_estimator.load_segmentation_model()
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass