import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
from rclpy.duration import Duration
import torch
import math
import numpy as np
import torchvision.transforms as T
from nav_msgs.msg import Odometry
from decord import VideoReader, cpu
from PIL import Image as PIL_Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from collections import deque
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import random

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class Planner(Node):
    def __init__(self):
        super().__init__('lvlm_node')

        # ROS 2 publishers/subscribers
        self.image_sub = self.create_subscription(Image, '/robot_1/sensors/front_stereo/left/image_rect', self.image_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/robot_1/odometry_conversion/odometry', self.odom_callback, 10)
        self.output_pub = self.create_publisher(Path, '/robot_1/global_plan', 10)
        self.prompt_sub = self.create_subscription(String, '/input_prompt', self.target_object_callback, 10)
        self.frame_memory = deque(maxlen=5)
        self.pose_memory = deque(maxlen=200)
        self.src2rdf_transform = self.mat_3x3_to_4x4(self.get_coord_system_transform("flu", "rdf"))
        self.bridge = CvBridge()
        self.create_timer(1.0, self.update_global_plan)
        self.create_timer(0.1, self.publish_current_path)
        self._target_objects = None
        self.waypoint_locked = False
        self.current_goal = None
        self.current_path_msg = None

        # Load InternVL3-2B model once
        self.get_logger().info("Loading internvl3-2B model...")
        self.model_path = "OpenGVLab/InternVL3-2B"
        self.device_map = self.split_model(self.model_path)
        self.model = AutoModel.from_pretrained(
                self.model_path, 
                torch_dtype=torch.bfloat16, 
                load_in_8bit=True, 
                low_cpu_mem_usage=True, 
                use_flash_attn=True, 
                trust_remote_code=True,
                device_map = self.device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=32, do_sample=True)
        self.get_logger().info("InternVL3-2B model loaded!")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            pil_image = PIL_Image.fromarray(cv_image)
            pixel_values = self.load_image(pil_image, max_num=1).to(torch.bfloat16).cuda()
            self.frame_memory.append(pixel_values)

        except Exception as e:
            self.get_logger().error(f"Image preprocessing failed: {e}")
            return 
    
    def target_object_callback(self, msg):
        targets = [t.strip().lower() for t in msg.data.split(",") if t.strip()]
        if not targets:
            self._target_objects = None
        else:
            self._target_objects = targets

    def vector3_to_numpy(self, msg, hom=False):
        if hom:
            return np.array([msg.x, msg.y, msg.z, 0])
        else:
            return np.array([msg.x, msg.y, msg.z])
    
    def quat_to_numpy(self, msg):
        return np.array([msg.x, msg.y, msg.z, msg.w])
    
    def quat_to_rot_matrix(self, q, scalar_first=False):
        q = np.array(q, dtype=np.float32)
        if not scalar_first:
            # reorder to w, x, y, z
            q = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        w, x, y, z = q

        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        return R

    def pose_to_numpy(self, ros_pose):
        pose_t = self.vector3_to_numpy(ros_pose.position)
        pose_q = self.quat_to_numpy(ros_pose.orientation)
        pose_R = self.quat_to_rot_matrix(pose_q, scalar_first=False)
        pose_Rt_3x4 = np.concatenate((pose_R, pose_t.reshape(3, 1)), axis=1)
        row = np.array([[0,0,0,1]])
        pose_Rt_4x4 = np.concatenate((pose_Rt_3x4, row), axis=-2)
        return pose_Rt_4x4
    
    def transform_pose_4x4(self, pose_4x4: np.ndarray, transform_mat_4x4: np.ndarray) -> np.ndarray:
        return transform_mat_4x4 @ pose_4x4 @ np.linalg.inv(transform_mat_4x4)
    
    def get_coord_system_transform(self, src: str, tgt: str) -> np.ndarray:
        axes = dict(r=0, l=0, u=1, d=1, f=2, b=2)
        T = np.zeros((3, 3), dtype=np.float32)

        for i, tgt_dir in enumerate(tgt.lower()):
            a = axes[tgt_dir]
            for j, src_dir in enumerate(src.lower()):
                b = axes[src_dir]
                if a == b:
                    sign = 1 if src_dir == tgt_dir else -1
                    T[i, j] = sign
                    break
        return T
    
    def mat_3x3_to_4x4(self, mat: np.ndarray) -> np.ndarray:
        # Ensure shape (..., 3, 3)
        mat = np.asarray(mat)
        if mat.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (...,3,3) matrix, got {mat.shape}")
        
        # Build zeros and bottom row
        zeros = np.zeros((*mat.shape[:-2], 3, 1), dtype=mat.dtype)
        mat_3x4 = np.concatenate((mat, zeros), axis=-1)
        return self.mat_3x4_to_4x4(mat_3x4)
    
    def mat_3x4_to_4x4(self, mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat)
        if mat.shape[-2:] != (3, 4):
            raise ValueError(f"Expected (...,3,4) matrix, got {mat.shape}")

        # Create the bottom row [0, 0, 0, 1] with correct batch shape
        batch_shape = mat.shape[:-2]
        row = np.zeros((*batch_shape, 1, 4), dtype=mat.dtype)
        row[..., 0, 3] = 1.0

        # Concatenate along the row axis (-2)
        return np.concatenate((mat, row), axis=-2)


    def odom_callback(self, msg: Odometry):
        try:
            src_pose_4x4 = self.pose_to_numpy(msg.pose.pose)

            pitch = 0.1745
            R_pitch = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ], dtype=np.float32)
            T_base_to_cam = np.eye(4, dtype=np.float32)
            T_base_to_cam[:3, :3] = R_pitch

            src_pose_4x4 = src_pose_4x4 @ T_base_to_cam

            # Transform into RDF/world frame
            rdf_pose_4x4 = self.transform_pose_4x4(
                src_pose_4x4, self.src2rdf_transform
            )

            # Save final pose
            self.pose_4x4 = rdf_pose_4x4

            # Extract (x, y, z) in custom coords
            cur_pose_np = np.array([
                float(rdf_pose_4x4[2, 3]),
                float(-rdf_pose_4x4[0, 3]),
                float(-rdf_pose_4x4[1, 3])
            ])

            self.pose_memory.append(cur_pose_np)

            self.cur_pose_np = cur_pose_np

            if self.waypoint_locked and self.current_goal is not None:
                dist = np.linalg.norm(self.current_goal - self.cur_pose_np)
                if dist < 2:
                    self.waypoint_locked = False
                    self.get_logger().info("waypoint reached, unlocking planner")
                
                # elif random.random() < 0.1:
                #     self.waypoint_locked = False
                #     self.get_logger().info("randomly unlocking planner for replanning")

        except Exception as e:
            self.get_logger().error(f"Odometry procecssing failed: {e}")
            return
    
    def sample_periodic_frames(self, memory, num_samples=5, step=10):
        if not memory:
            return []
        
        memory_len = len(memory)
        indices = [max(memory_len - 1 - i*step, 0) for i in range(num_samples)]
        indices = sorted(indices)
        return [memory[i] for i in indices]

    def publish_current_path(self):
        if self.current_path_msg is None:
            return
        
        self.current_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.output_pub.publish(self.current_path_msg)
    
    def update_global_plan(self):
        if self.waypoint_locked:
            return 
        
        if not self.frame_memory or not self.pose_memory:
            self.get_logger().warn("Not enough data to plan yet")
            return 
        
        recent_frames = self.sample_periodic_frames(self.frame_memory, num_samples=5, step=10)
        pixel_values = torch.cat(recent_frames, dim=0).to(torch.bfloat16).cuda()

        # num_context_poses = 10
        # stride = max(1,len(self.pose_memory)//num_context_poses)
        # recent_poses = list(self.pose_memory)[::stride][-num_context_poses:]

        # pose_context_str = "\n".join(
        #     f"{i}: {pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}  # {'latest' if i==0 else 'older'}"
        #     for i, pose in enumerate(reversed(recent_poses))
        # )

        # if len(recent_poses) > 1:
        #     oldest = recent_poses[0]
        #     newest = recent_poses[-1]
        #     motion_vector = [newest[i] - oldest[i] for i in range(3)]
        #     motion_str = f"Robot has been moving roughly in direction x={motion_vector[0]:.2f}, y={motion_vector[1]:.2f}, z={motion_vector[2]:.2f}."
        # else:
        #     motion_str = ""


        prompt = (
            "<image>\n"
            f"Find {self._target_objects}. "
            "Based on the current first-person view images, "
            "decide the next action for the robot. "
            "Choose only one of the following actions: 'move forward', 'turn left', 'turn right'. "
            "Return ONLY the action as plain text, no explanation."
        )
        self.get_logger().info(f"Prompt sent to model:\n{prompt}")

        try:
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                self.generation_config
            )
            self.get_logger().info(f"Model response:\n{response}")
        except Exception as e:
            self.get_logger().error(f"Model chat failed: {e}")
            response = ""
        
        action = response.strip().lower()

        cur = np.array(self.cur_pose_np, dtype=np.float32)
        z_height = max(cur[2], 3.0)

        forward_dist = 4.0

        turn_angle = np.pi/3.0
        
        lookback_idx = -10
        if len(self.pose_memory) >= abs(lookback_idx):
            prev = np.array(list(self.pose_memory)[lookback_idx], dtype=np.float32)
            heading_vec = cur - prev
            heading_vec[2] = 0  # ignore vertical
            norm = np.linalg.norm(heading_vec)
            if norm > 0:
                heading_unit = heading_vec / norm
            else:
                heading_unit = np.array([1.0, 0.0, 0.0])  # default forward
        else:
            heading_unit = np.array([1.0, 0.0, 0.0])

        
        # Rotate heading for left/right
        if action == "turn left":
            c, s = np.cos(turn_angle), np.sin(turn_angle)
            heading_unit[:2] = np.array([c * heading_unit[0] - s * heading_unit[1],
                                        s * heading_unit[0] + c * heading_unit[1]])
        elif action == "turn right":
            c, s = np.cos(-turn_angle), np.sin(-turn_angle)
            heading_unit[:2] = np.array([c * heading_unit[0] - s * heading_unit[1],
                                        s * heading_unit[0] + c * heading_unit[1]])
        

        goal = cur + heading_unit * forward_dist
        goal[2] = z_height

        extended_point = goal + heading_unit * forward_dist
        extended_point[2] = z_height

        planned_points = [goal, extended_point]

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for pt in planned_points:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = float(pt[0])
            pose_stamped.pose.position.y = float(pt[1])
            pose_stamped.pose.position.z = float(pt[2])
            pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)
        
        self.current_path_msg = path_msg
        self.waypoint_locked = True
        self.current_goal = planned_points[0]
        self.get_logger().info(f"Planned action '{action}' -> path published with 2 points.")


        # planned_points = []
        # for line in response.splitlines():
        #     try:
        #         x, y, z = [float(v) for v in line.strip().split(',')]
        #         planned_points.append(np.array([x, y, z], dtype=np.float32))
        #     except:
        #         continue

        # # If only one point, add a second point 5m further along the direction from current pose
        # if len(planned_points) == 1:
        #     goal = planned_points[0]
        #     cur = np.array(self.cur_pose_np, dtype=np.float32)
        #     direction = goal - cur
        #     norm = np.linalg.norm(direction)
        #     if norm > 0:
        #         direction_unit = direction / norm
        #         extended_point = goal + direction_unit * 2.0  # 2 meters ahead
        #         extended_point[2] = goal[2] #set same height
        #         planned_points.append(extended_point)

        #     # Build Path message
        #     path_msg = Path()
        #     path_msg.header.stamp = self.get_clock().now().to_msg()
        #     path_msg.header.frame_id = "map"

        #     for pt in planned_points:
        #         pose_stamped = PoseStamped()
        #         pose_stamped.header = path_msg.header
        #         pose_stamped.pose.position.x = float(pt[0])
        #         pose_stamped.pose.position.y = float(pt[1])
        #         pose_stamped.pose.position.z = float(pt[2])
        #         # Optionally: set orientation if known
        #         pose_stamped.pose.orientation.w = 1.0
        #         path_msg.poses.append(pose_stamped)

        #     # Publish
        #     #self.output_pub.publish(path_msg)
        #     #self.get_logger().info(f"Published planned path with {len(path_msg.poses)} points.")
        #     self.current_path_msg = path_msg
        #     self.waypoint_locked = True
        #     self.current_goal = planned_points[0]




            
        #     prompt = (f'<image>\nFind {self._target_objects}. ' 
        #               f'List three unique objects or areas that are most helpful as clues or context to locate the {self._target_objects}. ' 
        #               f'Write ONLY the object or area names as a plain comma-separated list.')
        #     response = self.model.chat(self.tokenizer, pixel_values, prompt, self.generation_config)
        #     print(f'User: {prompt}\nAssistant: {response}')

        #     #Extract only the part after ASSISTANT:
        #     objects_text = response.strip()
        #     print("output_text: ", objects_text)

        #     # Publish
        #     self.output_pub.publish(String(data=objects_text))

        #     self.last_call = now
        #     self.get_logger().info("Published LVLM output")

        #     self.trigger_active = False
        # else:
        #     return

    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.model.rotary_emb'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map


    def load_image(self, image, input_size=448, max_num=1):
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values


def main(args=None):
    rclpy.init(args=args)
    node = Planner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

