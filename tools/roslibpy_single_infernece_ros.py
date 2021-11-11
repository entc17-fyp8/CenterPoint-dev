
# import rospy
# import ros_numpy
import numpy as np
import copy
import json
import os
import sys
import torch
import time 
import roslibpy
import base64

# from std_msgs.msg import Header
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import PointCloud2, PointField
# from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion

# from det3d import __version__, torchie
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator

############################################## To make the message types available

# Header = {'seq' : 0, 'stamp' : 0, 'frame_id' : ''}

# PointField = {'INT8':1, 'UINT8' : 2, 'INT16' : 3, 'UINT16' : 4, 'INT32' : 5, 'UINT32' : 6, 'FLOAT32' : 7, 'FLOAT64' : 8, 'name' : '', 'offset' : 0, 'datatype' : 0, 'count' : 0}

# PointCloud2 = {'header' : Header, 'height' : 0, 'width' : 0, 'fields' : PointField, 'is_bigendian' : False, 'point_step' : 0, 'row_step' : 0, 'data' : [], 'is_dense' : False}

# BoundingBox = {'header' : Header, 'pose' :{'position' : {'x' : 0, 'y' : 0, 'z' : 0}, 'orientation' : {'x' : 0, 'y' : 0, 'z' : 0, 'w' : 0}}, 'vector' : {'x' : 0, 'y' : 0, 'z' : 0}, 'value' : 0, 'label' : 0}

# BoundingBoxArray = {'header' : Header, 'boundingbox' : BoundingBox}

class PointField():
    INT8 = 1
    UINT8 = 2
    INT16= 3
    UINT16 = 4
    INT32 = 5
    UINT32 = 6
    FLOAT32 = 7
    FLOAT64 = 8
    name = ''
    offset = 0
    datatype = 0
    count = 0

class PointCloud2():
    class header():
        seq = 0
        stamp = 0
        frame_id = ''
    height = 0
    width = 0
    class fields():
        INT8 = 1
        UINT8 = 2   
        INT16= 3
        UINT16 = 4
        INT32 = 5
        UINT32 = 6
        FLOAT32 = 7
        FLOAT64 = 8
        name = ''
        offset = 0
        datatype = 0
        count = 0
    is_bigendian = False
    point_step = 0
    row_step = 0
    data = []
    is_dense = False

class BoundingBox():
    class header():
        seq = 0
        stamp = 0
        frame_id = ''
    class pose():
        class position():
            x = 0
            y = 0
            z = 0
        class orientation():
            x = 0
            y = 0
            z = 0
            w = 0
    class vector():
        x = 0
        y = 0
        z = 0
    value = 0
    label = 0

class BoundingBoxArray():
    class header():
        seq = 0
        stamp = 0
        frame_id = ''
    class BoundingBox():
        class header():
            seq = 0
            stamp = 0
            frame_id = ''
        class pose():
            class position():
                x = 0
                y = 0
                z = 0
            class orientation():
                x = 0
                y = 0
                z = 0
                w = 0
        class vector():
            x = 0
            y = 0
            z = 0
        value = 0
        label = 0
###################################################################################

def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)

def get_annotations_indices(types, thresh, label_preds, scores):
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices  


def remove_low_score_nu(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()
    
    car_indices =                  get_annotations_indices(0, 0.4, label_preds_, scores_)
    truck_indices =                get_annotations_indices(1, 0.4, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(2, 0.4, label_preds_, scores_)
    bus_indices =                  get_annotations_indices(3, 0.3, label_preds_, scores_)
    trailer_indices =              get_annotations_indices(4, 0.4, label_preds_, scores_)
    barrier_indices =              get_annotations_indices(5, 0.4, label_preds_, scores_)
    motorcycle_indices =           get_annotations_indices(6, 0.15, label_preds_, scores_)
    bicycle_indices =              get_annotations_indices(7, 0.15, label_preds_, scores_)
    pedestrain_indices =           get_annotations_indices(8, 0.1, label_preds_, scores_)
    traffic_cone_indices =         get_annotations_indices(9, 0.1, label_preds_, scores_)
    
    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices + 
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])

    return img_filtered_annotations


class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        
    def initialize(self):
        self.read_config()
        
    def read_config(self):
        config_path = self.config_path
        cfg = Config.fromfile(self.config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        self.net.load_state_dict(torch.load(self.model_path)["state_dict"])
        self.net = self.net.to(self.device).eval()

        self.range = cfg.voxel_generator.range
        self.voxel_size = cfg.voxel_generator.voxel_size
        self.max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = cfg.voxel_generator.max_voxel_num
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[1],
        )

    def run(self, points):
        t_t = time.time()
        print(f"input points shape: {points.shape}")
        num_features = 5        
        self.points = points.reshape([-1, num_features])
        self.points[:, 4] = 0 # timestamp value 
        
        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)
        
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)
        
        self.inputs = dict(
            voxels = voxels,
            num_points = num_points,
            num_voxels = num_voxels,
            coordinates = coords,
            shape = [grid_size]
        )
        torch.cuda.synchronize()
        t = time.time()

        with torch.no_grad():
            outputs = self.net(self.inputs, return_loss=False)[0]
    
        # print(f"output: {outputs}")
        
        torch.cuda.synchronize()
        print("  network predict time cost:", time.time() - t)

        outputs = remove_low_score_nu(outputs, 0.45)

        boxes_lidar = outputs["box3d_lidar"].detach().cpu().numpy()
        print("  predict boxes:", boxes_lidar.shape)

        scores = outputs["scores"].detach().cpu().numpy()
        types = outputs["label_preds"].detach().cpu().numpy()

        boxes_lidar[:, -1] = -boxes_lidar[:, -1] - np.pi / 2

        print(f"  total cost time: {time.time() - t_t}")

        return scores, boxes_lidar, types

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''
    '''
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    return points

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

def rslidar_callback(msg):
    print("Callback was called")
    t_t = time.time()
    arr_bbox = BoundingBoxArray()

    # msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    # msg_cloud = pointcloud2_to_array(msg)   
    msg_cloud = pointcloud2_to_array_obj(msg)

    np_p = get_xyz_points(msg_cloud, True)
    print("  ")
    scores, dt_box_lidar, types = proc_1.run(np_p)

    if scores.size != 0:
        for i in range(scores.size):
            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            # bbox.header.stamp = rospy.Time.now()
            bbox.header.stamp = roslibpy.get_time()
            q = yaw2quaternion(float(dt_box_lidar[i][8]))
            bbox.pose.orientation.x = q[1]
            bbox.pose.orientation.y = q[2]
            bbox.pose.orientation.z = q[3]
            bbox.pose.orientation.w = q[0]           
            bbox.pose.position.x = float(dt_box_lidar[i][0])
            bbox.pose.position.y = float(dt_box_lidar[i][1])
            bbox.pose.position.z = float(dt_box_lidar[i][2])
            bbox.dimensions.x = float(dt_box_lidar[i][4])
            bbox.dimensions.y = float(dt_box_lidar[i][3])
            bbox.dimensions.z = float(dt_box_lidar[i][5])
            bbox.value = scores[i]
            bbox.label = int(types[i])
            arr_bbox.boxes.append(bbox)
    print("total callback time: ", time.time() - t_t)
    arr_bbox.header.frame_id = msg.header.frame_id
    arr_bbox.header.stamp = msg.header.stamp
    if len(arr_bbox.boxes) is not 0:
        pub_arr_bbox.publish(arr_bbox)
        arr_bbox.boxes = []
    else:
        arr_bbox.boxes = []
        pub_arr_bbox.publish(arr_bbox)
   
def pointcloud2_to_array(cloud_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray 
    
    Reshapes the returned array to have shape (height, width), even if the height is 1.
    The reason for using np.frombuffer rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]
    
    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

def pointcloud2_to_array_obj(cloud_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray 
    
    Reshapes the returned array to have shape (height, width), even if the height is 1.
    The reason for using np.frombuffer rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype_obj(cloud_msg["fields"], cloud_msg["point_step"])

    # parse the cloud into an array
    # t = base64.decodebytes(cloud_msg["data"])
    t = base64.b64decode(cloud_msg["data"].encode('utf-8'))
    cloud_arr = np.frombuffer(t, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]
    
    if squeeze and cloud_msg["height"] == 1:
        return np.reshape(cloud_arr, (cloud_msg["width"],))
    else:
        return np.reshape(cloud_arr, (cloud_msg["height"], cloud_msg["width"]))

############# Copied from (https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/point_cloud2.py) ############

type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

pftype_to_nptype = dict(type_mappings)

DUMMY_FIELD_PREFIX = '__'

def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

def fields_to_dtype_obj(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f["offset"]:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f["datatype"]]
        if f["count"] != 1:
            dtype = np.dtype((dtype, f["count"]))

        np_dtype_list.append((f["name"], dtype))
        offset += pftype_sizes[f["datatype"]] * f["count"]

###################################################################################################

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1
        
    return np_dtype_list

def just_print(msg):
    print("msg_received")

if __name__ == "__main__":

    global proc
    ## CenterPoint
    config_path = '/workspace/CenterPoint/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py'
    # model_path = 'det3d/models/last.pth'
    model_path = '/workspace/Checkpoints/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/epoch_20.pth'

    proc_1 = Processor_ROS(config_path, model_path)
    
    proc_1.initialize()
    
    # rospy.init_node('centerpoint_ros_node')
    # No initialization of nodes in roslibpy
    # centerpoint_ros_node = roslibpy.Topic() # ???
    # sub_lidar_topic = [ "/velodyne_points", 
                        # "/top/rslidar_points",
                        # "/points_raw", 
                        # "/lidar_protector/merged_cloud", 
                        # "/merged_cloud",
                        # "/lidar_top", 
                        # "/roi_pclouds"]
    
    # sub_ = rospy.Subscriber(sub_lidar_topic[5], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)
    client = roslibpy.Ros(host='localhost', port=9090)    # host and port??? - this is how another ros environment is connected
    client.run()
    sub_ = roslibpy.Topic(ros = client, name = "/lidar/top", message_type = 'sensor_msgs/PointCloud2') # Callback function???
    sub_.subscribe(rslidar_callback)

    # sub_2 = roslibpy.Topic(ros = client, name = "/camera/front_right/rgb_image", message_type = 'sensor_msgs/Image') # Callback function???
    # sub_2.subscribe(just_print)

    # pub_arr_bbox = rospy.Publisher("pp_boxes", BoundingBoxArray, queue_size=1)
    pub_arr_bbox = roslibpy.Topic(ros = client, name = "/pp_boxes", message_type = 'jsk_recognition_msgs/BoundingBoxArray', queue_size=1)

    print("[+] CenterPoint ros_node has started!")    
    # rospy.spin()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        client.terminate()