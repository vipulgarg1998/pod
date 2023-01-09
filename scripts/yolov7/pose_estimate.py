#!/usr/bin/env python3
# ROS Libs
import rospy
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from std_msgs.msg import Header, Empty
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from jsk_rviz_plugins.msg import OverlayText
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

from pose_mimic.srv import Objects

# Python Libs
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
    def __init__(self, model_filename = '../../weights/yolov7-w6-pose.pt'):
        # super().__init__('minimal_publisher')

        self.get_objects_srv = '/yolo/segmentation/objects'

        tss = ApproximateTimeSynchronizer(
            [Subscriber("zed/zed_node/rgb/image_rect_color", Image), 
            Subscriber("zed/zed_node/depth/depth_registered", Image)], 10, 1)
        tss.registerCallback(self.rgbd_callback)
        
        # self.image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.image_callback)
        self.keypoint_pub = rospy.Publisher('keypoints', PointCloud2, queue_size=10)
        self.stop_segmentation_pub = rospy.Publisher('/yolo/segmentation/stop', Empty, queue_size=10)
        self.vector_pub = rospy.Publisher("visualization_marker", Marker, queue_size=0)
        self.pointed_obj_pub = rospy.Publisher("pointed_object", OverlayText, queue_size=0)
        self.camera_info_sub = rospy.Subscriber("/zed/zed_node/left/camera_info", CameraInfo, self.camera_info_callback)

        # For OpenCV
        self.cv_bridge = CvBridge()
        self.cv_image = None

        # For Yolo 
        self.model_filename = model_filename
        self.device = None
        self.weights = None
        self.model = None

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

        # For Segmentation
        self.objects = None


    def load_model(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weigths = torch.load(self.model_filename)
        self.model = self.weigths['model']
        self.model = self.model.half().to(self.device)
        _ = self.model.eval()
        print("Model Loaded")

    def camera_info_callback(self, camera_info_msg):
        K = camera_info_msg.K
        self.fx = float(K[0])
        self.cx = float(K[2])
        self.fy = float(K[4])
        self.cy = float(K[5])
        self.width = int(camera_info_msg.width)
        self.height = int(camera_info_msg.height)
        self.caliberation_params_init = True

    def rgbd_callback(self, image_msg, depth_image_msg):
        if(self.model == None or not self.caliberation_params_init):
            return
        cv_depth_image = self.cv_bridge.imgmsg_to_cv2(depth_image_msg)

        keypoints_2d, ratio = self.image_callback(image_msg, view = False)
        keypoints_3d = self.cvt_kpts_2d_to_3d(keypoints_2d, ratio, cv_depth_image, self.fx, self.fy, self.cx, self.cy, view=False)
        if(len(keypoints_3d) == 0):
            return
        directional_vector = self.get_right_hand_direction_vector(keypoints_3d)

        keypoints_3d = np.array(keypoints_3d)
        sensor_msg = self.to_sensor_msgs(keypoints_3d)
        self.keypoint_pub.publish(sensor_msg)
        if(directional_vector != None):
            self.draw_vector(directional_vector)
            if(self.objects != None):
                self.highlight_object(directional_vector)
        # print("Direction Vector is ", directional_vector)


    def to_sensor_msgs(self, keypoints_3d):
        points_3d = keypoints_3d[:, 3:]
        # Add intensity = 1 to the 4th column
        points_3d = np.hstack((points_3d,np.ones([points_3d.shape[0],1], points_3d.dtype)))
        # print(points_3d)
        header = Header()
        header.frame_id = self.frame_id
        header.stamp = rospy.Time.now()

        points_3d = points_3d
        # print(points_3d)

        pc2 = point_cloud2.create_cloud(header, self.fields, points_3d)
        return pc2

    def get_o3d_pcl(self, cv_depth_image, fx, fy, cx, cy):
        pcl = o3d.geometry.PointCloud()
        points = []
        for u in range(cv_depth_image.rows):
            for v in range(cv_depth_image.cols):
                z = cv_depth_image[u, v]
                x = (v - cx)*z/fx
                y = (u - cy)*z/fy

                point = [x, y, z]
                points.append(point)

        pcl.points = o3d.utility.Vector3dVector(np.array(points))
        o3d.visualization.draw_geometries([pcl])


    def cvt_kpts_2d_to_3d(self, keypoints_2d, ratio, cv_depth_image, fx, fy, cx, cy, view = True):
        steps = 3
        data = []
        im = cv_depth_image.copy()
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for idx in range(keypoints_2d.shape[0]): # For each person
            kpts = keypoints_2d[idx, 7:]
            num_keypoints = len(kpts)//steps
            for i in range(num_keypoints):
                x_coord, y_coord = kpts[steps*i], kpts[steps*i + 1]
                if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                    if steps == 3:
                        conf = kpts[steps * i + 2]
                        if conf > 0.5:
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
                            data_point = [idx, i, conf, x, y, z]
                            data.append(data_point)

                            if(view):
                                cv2.circle(im, (int(x_coord_original), int(y_coord_original)), 5, (int(255), int(255), int(255)), -1)
                            # print("ID ", i, "3D point is ", "x: ", x, " y: ", y, " z: ", z)

        if(view):
            cv2.imshow('depth image', im)
            cv2.waitKey(1)
        return data

    def get_right_hand_direction_vector(self, data):
        # "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", 
        # "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", 
        # "right_knee", "left_ankle", "right_ankle"
        wrist_coord = np.nan
        elbow_coord = np.nan
        for data_point in data:
            if(data_point[1] == 8):
                elbow_coord = data_point[3:]
                # print(" Elbow Coord: ", elbow_coord)
            if(data_point[1] == 10):
                wrist_coord = data_point[3:]
                # print(" Wrist Coord: ", wrist_coord)
        
        if(not (np.isfinite(elbow_coord).all() and np.isfinite(wrist_coord).all())):
            return None
        else:
            # return self.get_vector(wrist_coord, elbow_coord)
            return [elbow_coord, wrist_coord]

    def get_vector(self, point_a, point_b):
        vector = []
        for i in range(len(point_a)):
            vector.append(point_a[i] - point_b[i])

        return vector

    def preprocess_images(self, image, new_shape = (640, 640)):
        return cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)

    def image_callback(self, image_msg, view = True):
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
        # cv_image = self.preprocess_images(cv_image, new_shape=(640, 640))
        tensor_image, ratio = self.convert_image(cv_image)
        keypoints = self.get_keypoints_2d(tensor_image, view = view)
        return keypoints, ratio

    def convert_image(self, image):
        image, ratio, _ = self.letterbox(image, stride=64, auto=False, scaleFill = True)
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        image = image.to(self.device)
        image = image.half()
        return image, ratio

    def get_dist(self, vector_p1, vector_p2, point):
        vector_p1 = np.array(vector_p1)
        vector_p2 = np.array(vector_p2)
        point = np.array(point)
        return np.linalg.norm(np.cross(vector_p2-vector_p1, vector_p1-point))/np.linalg.norm(vector_p2-vector_p1)

    def highlight_object(self, directional_vector):
        min_dist = 1000
        obj_pointed = None
        for i, label in enumerate(self.objects.labels):
            obj_centroid = [self.objects.x[i], self.objects.y[i], self.objects.z[i]] 
            dist = self.get_dist(directional_vector[0], directional_vector[1], obj_centroid)
            if(dist < min_dist):
                min_dist = dist
                obj_pointed = label

        text = OverlayText()
        text.text = f"Pointed Object is {obj_pointed}"
        self.pointed_obj_pub.publish(text)
        print(text.text)


    def get_objects(self):
        rospy.wait_for_service(self.get_objects_srv)
        repeat_req = True
        while(repeat_req):
            try:
                get_objects = rospy.ServiceProxy(self.get_objects_srv, Objects)
                objects = get_objects()
                repeat_req = len(objects.labels) == 0

                if(not repeat_req):
                    self.objects = objects
                    print(self.objects)
                    # self.stop_segmentation_pub.publish(Empty())
                # print(objects.labels)
                # print(objects.x)
                # print(objects.y)
                # print(objects.z)
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

        self.load_model()

    def get_keypoints_2d(self, image, view = True):
        # Forward Pass to the model
        start_time = time.time()            # Get the start time.
        with torch.no_grad():
            output, _ = self.model(image)
        end_time = time.time()              # Get the end time.
        fps = 1 / (end_time - start_time)   # Get the fps.

        # Get keypoints
        output = self.non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
        output = self.output_to_keypoint(output)
        # print(output)
        if(view):
            # Plot the keypoints
            nimg = image[0].permute(1, 2, 0) * 255
            nimg = nimg.cpu().numpy().astype(np.uint8)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
            
            for idx in range(output.shape[0]):
                self.plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        
                # Comment/Uncomment the following lines to show bounding boxes around persons.
                xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
                xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
                cv2.rectangle(
                    nimg,
                    (int(xmin), int(ymin)),
                    (int(xmax), int(ymax)),
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
        
            # Write the FPS on the current frame.
            cv2.putText(nimg, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            # Convert from BGR to RGB color format.
            cv2.imshow('image', nimg)
            #   out.write(nimg)
            # Press `q` to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.exit()
        return output

    def exit(self):
        cv2.destroyAllWindows()

    def draw_vector(self, vector, scale = 4, hide = False):
        arrow = Marker()
        arrow.header.frame_id = self.frame_id
        arrow.header.stamp = rospy.Time.now()
        arrow.id = 0
        arrow.type = Marker.ARROW
        start_point = Point()
        start_point.x = vector[0][0]
        start_point.y = vector[0][1]
        start_point.z = vector[0][2]
        end_point = Point()
        end_point.x = scale*vector[1][0] - (scale-1)*start_point.x
        end_point.y = scale*vector[1][1] - (scale-1)*start_point.y
        end_point.z = scale*vector[1][2] - (scale-1)*start_point.z
        arrow.points = [start_point, end_point]
        arrow.scale.x = 0.1
        arrow.scale.y = 0.1
        arrow.scale.z = 0.05
        arrow.color.a = 1.0
        arrow.color.r = 1.0
        arrow.color.g = 1.0
        arrow.color.b = 1.0
        arrow.action = 2 if hide else 0
        self.vector_pub.publish(arrow)

    def plot_skeleton_kpts(self, im, kpts, steps, orig_shape=None):
        #Plot the skeleton and keypointsfor coco datatset
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                            [255, 255, 255]])

        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        radius = 5
        num_kpts = len(kpts) // steps

        for kid in range(num_kpts):
            r, g, b = pose_kpt_color[kid]
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                if steps == 3:
                    conf = kpts[steps * kid + 2]
                    if conf < 0.5:
                        continue
                cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

        for sk_id, sk in enumerate(skeleton):
            r, g, b = pose_limb_color[sk_id]
            pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
            pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
            if steps == 3:
                conf1 = kpts[(sk[0]-1)*steps+2]
                conf2 = kpts[(sk[1]-1)*steps+2]
                if conf1<0.5 or conf2<0.5:
                    continue
            if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
                continue
            if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
                continue
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

    def output_to_keypoint(self, output):
        # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
        targets = []
        for i, o in enumerate(output):
            kpts = o[:,6:]
            o = o[:,:6]
            for index, (*box, conf, cls) in enumerate(o.detach().cpu().numpy()):
                targets.append([i, cls, *list(*self.xyxy2xywh(np.array(box)[None])), conf, *list(kpts.detach().cpu().numpy()[index])])
        return np.array(targets)


    def non_max_suppression_kpt(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                            labels=(), kpt_label=False, nc=None, nkpt=None):
        """Runs Non-Maximum Suppression (NMS) on inference results
        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        if nc is None:
            nc = prediction.shape[2] - 5  if not kpt_label else prediction.shape[2] - 56 # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0,6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:5+nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                if not kpt_label:
                    conf, j = x[:, 5:].max(1, keepdim=True)
                    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
                else:
                    kpts = x[:, 6:]
                    conf, j = x[:, 5:6].max(1, keepdim=True)
                    x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]


            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output

    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        # print("New Unpad ", new_unpad)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)




def main(args=None):
    
    # initializing the subscriber node
    rospy.init_node('pose_estimate', anonymous=True)

    pose_estimator = PoseEstimator()
    pose_estimator.get_objects()
    # pose_estimator.load_model()
    # pose_estimator.load_segmentation_model()
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass