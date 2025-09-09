import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray
import json
import os
import numpy as np
from nav_msgs.msg import Odometry
import itertools

class MissionChecker(Node):
    def __init__(self):
        super().__init__('mission_checker')
        #AbondonedCity
        #self.annotation_file = "/app/annotations/transformed_annotations/AbandonedCity_t_x0_y0_z0_o_x0_y0_z0.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/AbandonedCity_t_x0_y80_z0_o_x0_y0_z-90.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/AbandonedCity_t_x5_y-60_z0_o_x0_y0_z90.json"
        
        #AbandonedFactory
        #self.annotation_file = "/app/annotations/transformed_annotations/AbandonedFactory_t_x0_y0_z0.5_o_x0_y0_z0.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/AbandonedFactory_t_x-5_y35_z0.5_o_x0_y0_z-90.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/AbandonedFactory_t_x-5_y-15_z0.5_o_x0_y0_z90.json"
        
        #ConstructionSite
        #self.annotation_file = "/app/annotations/transformed_annotations/ConstructionSite_t_x-27_y-8.5_z0.2_o_x0_y0_z0.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/ConstructionSite_t_x60_y-3_z0.2_o_x0_y0_z-90.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/ConstructionSite_t_x48_y-39_z0.2_o_x0_y0_z90.json"
        
        #DowntownWest
        #self.annotation_file = "/app/annotations/transformed_annotations/DowntownWest_t_x0_y0_z0_o_x0_y0_z0.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/DowntownWest_t_x2_y-60_z0_o_x0_y0_z90.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/DowntownWest_t_x-120_y0_z0_o_x0_y0_z0.json"
        
        #FireAcademy
        #self.annotation_file = "/app/annotations/transformed_annotations/FireAcademy_t_x0_y0_z0_o_x0_y0_z0.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/FireAcademy_t_x-15_y0_z0_o_x0_y0_z-90.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/FireAcademy_t_x30_y30_z1_o_x0_y0_z-90.json"
        
        #MilitaryBase
        #self.annotation_file = "/app/annotations/transformed_annotations/MilitaryBase_t_x1100_y200_z0_o_x0_y0_z90.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/MilitaryBase_t_x1070_y300_z0_o_x0_y0_z-90.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/MilitaryBase_t_x1114_y28_z0_o_x0_y0_z90.json"
        
        #ModernCityDowntown
        #self.annotation_file = "/app/annotations/transformed_annotations/ModernCityDowntown_t_x-4_y20_z0.2_o_x0_y0_z-90.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/ModernCityDowntown_t_x36_y-80_z0.2_o_x0_y0_z90.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/ModernCityDowntown_t_x-17_y-69_z0.2_o_x0_y0_z0.json"
        
        #Neighborhood
        self.annotation_file = "/app/annotations/transformed_annotations/Neighborhood_t_x0_y0_z0_o_x0_y0_z0.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/Neighborhood_t_x-20_y-80_z0_o_x0_y0_z90.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/Neighborhood_t_x160_y-19_z0_o_x0_y0_z180.json"
        
        #SnowyVillage
        #self.annotation_file = "/app/annotations/transformed_annotations/SnowyVillage_t_x-152_y-80_z-2_o_x0_y0_z90.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/SnowyVillage_t_x-145_y20_z-2.5_o_x0_y0_z-90.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/SnowyVillage_t_x-200_y-80_z-2_o_x0_y0_z0.json"
        
        #Shipyard
        #self.annotation_file = "/app/annotations/transformed_annotations/Shipyard_t_x0_y-100_z-0.5_o_x0_y0_z0.json"
        #self.annotation_file = "/app/annotations/transformed_annotations/Shipyard_t_x55_y50_z-0.5_o_x0_y0_z-90.json"
        #self.annotation_file="/app/annotations/transformed_annotations/Shipyard_t_x70_y-100_z-0.5_o_x0_y0_z90.json"

        self.annotations = self.load_annotations(self.annotation_file)
        
        self.gt_bbox_publisher = self.create_publisher(MarkerArray, 'annotation_bboxes', 10)
        self.trajectory_sub = self.create_subscription(MarkerArray, '/robot_1/trajectory_controller/trajectory_vis', self.trajectory_callback, 10)
        self.task_sub = self.create_subscription(String, '/input_prompt', self.task_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/robot_1/odometry_conversion/odometry', self.odom_callback, 10)

        self.traj_length_pub = self.create_publisher(String, "trajectory_length", 10)
        self.success_count_pub = self.create_publisher(String, "success_count", 10)
        self.progress_pub = self.create_publisher(String, "progress", 10)
        self.ppl_pub = self.create_publisher(String, "ppl", 10)
        self.latest_trajectory = None
        self.timer = self.create_timer(1.0, self.publish_annotations)

        self.task = None
        self.traj_length = 0.0
        self.success_count = 0
        self.progress = 0.0
        self.ppl = 0.0
        self.visited_objects = set()
        self.target_objects = []
        self.target_annotations = [] #[ann for ann in self.annotations if ann.get("class") in self.target_objects]
        self.oracle_path_lengths = None

        self.starting_time = None
        self.odom_time = None

        self.method = 'FPV'


    def load_annotations(self, file_path):
        if not os.path.exists(file_path):
            self.get_logger().error(f"Annotation file not found: {file_path}")
            return {}
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def task_callback(self, msg):
        if self.task is None:
            self.task = msg.data
            self.target_objects = self.task.split(", ")
            self.target_annotations = [ann for ann in self.annotations if ann.get("class") in self.target_objects]
            print("self.target_annotations", self.target_annotations)
            self.read_oracle_paths()

    def read_oracle_paths(self):
        base = os.path.splitext(os.path.basename(self.annotation_file))[0]
        if self.task is not None and len(self.task) > 0:
            filename = f"{base}_{self.task}.json"
            o_filepath = os.path.join("/app/annotations/oracle_paths", filename)
            with open(o_filepath, 'r') as f:
                data = json.load(f)
            self.oracle_path_lengths = {int(k): v for k,v in data.items()}

    def odom_callback(self, msg):
        stamp = msg.header.stamp
        self.odom_time  = stamp.sec + stamp.nanosec * 1e-9

    def set_starting_time(self):
        if self.starting_time is None and self.odom_time is not None:
            if self.traj_length > 5.0:
                self.starting_time = self.odom_time

    def trajectory_callback(self, msg):
        ###compute trajectory length and publish it        
        traj_np = np.array([[p.y, -p.x, p.z] for p in msg.markers[1].points])
        self.latest_trajectory = traj_np
        self.get_logger().info(f"Received trajectory with {len(msg.markers)} markers") 
        if traj_np.shape[0] < 2:
            self.traj_length = 0.0
        else:
            diffs = np.diff(traj_np, axis=0)
            segment_lengths = np.linalg.norm(diffs, axis=1)
            self.traj_length = np.round(segment_lengths.sum(),2)
        print(f"Trajectory length: {self.traj_length}")
        if self.starting_time is not None and self.odom_time is not None:
            elapsed = round(self.odom_time - self.starting_time, 2) 
            print(f"Traveled for {elapsed} sec")

            if int(elapsed) % 10 == 0:
                base = os.path.splitext(os.path.basename(self.annotation_file))[0]
                if self.task is not None and len(self.task) > 0:
                    rfilename = f"{base}_{self.task}_{self.method}.txt"
                    result_file_path = os.path.join("/app/results", rfilename)
                    with open(result_file_path, 'a') as rf:
                        rf.write(f"Time,{int(elapsed)},Progress,{round(self.progress*100,2)},PPL,{round(self.ppl*100,2)}\n")
                        rf.flush()

        traj_msg = String()
        traj_msg.data = str(self.traj_length)
        self.traj_length_pub.publish(traj_msg)

        self.set_starting_time()

        ###check if trajectory reached target objects
        reached_count = 0
        if traj_np.size > 0:
            for ann in self.target_annotations:
                center = np.array(ann["bbox_world"]["center_xyz_m"])
                size = np.array(ann["bbox_world"]["size_xyz_m"])

                half_extents = size / 2.0 + 4.0

                min_corner = center - half_extents
                max_corner = center + half_extents
                inside_mask = np.all((traj_np >= min_corner) & (traj_np <= max_corner), axis=1)
                if np.any(inside_mask):
                    reached_count += 1
        else:
            self.get_logger().warn("Received empty trajectory, skipping check.")

        self.success_count = reached_count
        if len(self.target_annotations) > 0:
            self.progress = float(self.success_count / len(self.target_annotations))
            if self.oracle_path_lengths is not None:
                if self.success_count > 0:
                    self.ppl = (self.oracle_path_lengths[self.success_count]/self.traj_length) * self.progress

        success_count_msg = String()
        success_count_msg.data = str(int(self.success_count))
        self.success_count_pub.publish(success_count_msg)

        progress_msg = String()
        progress_msg.data = str(round(self.progress*100,2))
        self.progress_pub.publish(progress_msg)

        ppl_msg = String()
        ppl_msg.data = str(round(self.ppl*100,2))
        self.ppl_pub.publish(ppl_msg)

    def publish_annotations(self):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i, ann in enumerate(self.target_annotations):
            bbox = ann.get('bbox_world', {})
            center = bbox.get('center_xyz_m', [0,0,0])
            size = bbox.get('size_xyz_m', [1,1,1])

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = now
            marker.ns = 'annotation_cuboids'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = -center[1]
            marker.pose.position.y = center[0]
            marker.pose.position.z = center[2]
            marker.scale.x = size[1]
            marker.scale.y = size[0]
            marker.scale.z = size[2]
            marker.color = ColorRGBA(r=0.5, g=0.5, b=0.0, a=0.3)
            marker.lifetime.sec = 1
            marker_array.markers.append(marker)
        self.gt_bbox_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = MissionChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()