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
from decord import VideoReader, cpu
from PIL import Image as PIL_Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class LVLMNode(Node):
    def __init__(self):
        super().__init__('lvlm_node')

        # ROS 2 publishers/subscribers
        self.image_sub = self.create_subscription(Image, '/robot_1/sensors/front_stereo/left/image_rect', self.image_callback, 10)
        self.trigger_sub = self.create_subscription(Bool, '/lvlm_trigger', self.trigger_callback, 10)
        self.prompt_sub = self.create_subscription(String, '/input_prompt', self.target_object_callback, 10)
        self.output_pub = self.create_publisher(String, '/lvlm_output', 10)
        
        self.bridge = CvBridge()
        self.trigger_active = False
        self._target_object = None

        # Throttle frequency (optional)
        self.last_call = self.get_clock().now()
        self.interval = Duration(seconds=60)

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

    def trigger_callback(self, msg: Bool):
        #self.get_logger().info(f"Received trigger: {msg.data}")
        self.trigger_active = msg.data
    
    def target_object_callback(self, msg):
        data_cleaned = msg.data.strip().lower()
        if data_cleaned == "":
            self._target_object = None
        else:
            self._target_object = data_cleaned
        #print("self._target_object", self._target_object)

    def image_callback(self, msg: Image):
        if not self.trigger_active:
            return 
        
        now = self.get_clock().now()
        if (now - self.last_call) < self.interval:
            return  # skip if called too soon

        if self._target_object is not None:
            # prompt = (
            #     f"USER: <image>\n"
            #     f"Find {self._target_object} in this scene. "
            #     f"List three unique objects or areas that are most helpful as clues or context to locate the {self._target_object}. "
            #     f"Do NOT include the {self._target_object} itself, its name, or variations of its name. "
            #     f"Write ONLY the object or area names as a plain comma-separated list in lowercase. "
            #     f"Do not use 'a', 'an', or 'the'. Do not add any extra words.\n"
            #     "ASSISTANT:")

            # print("prompt: ", prompt)

            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            pil_image = PIL_Image.fromarray(cv_image) 

            pixel_values = self.load_image(pil_image, max_num=1).to(torch.bfloat16).cuda()
            prompt = (f'<image>\nFind {self._target_object}. ' 
                      f'List three unique objects or areas that are most helpful as clues or context to locate the {self._target_object}. ' 
                      f'Write ONLY the object or area names as a plain comma-separated list.')
            response = self.model.chat(self.tokenizer, pixel_values, prompt, self.generation_config)
            print(f'User: {prompt}\nAssistant: {response}')
            # Preprocess and generate
            #inputs = self.processor(text=prompt, images=cv_image, return_tensors="pt").to("cuda")
            #for k,v in inputs.items():
            #    print(k,v.shape)
            #output = self.model.generate(**inputs, max_new_tokens=20)
            #generated_text = self.processor.batch_decode(output, skip_special_tokens=True)

            #Extract only the part after ASSISTANT:
            objects_text = response.strip()
            print("output_text: ", objects_text)

            # Publish
            self.output_pub.publish(String(data=objects_text))

            self.last_call = now
            self.get_logger().info("Published LVLM output")

            self.trigger_active = False
        else:
            return

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
            # split the image
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
        #image = PIL_Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values


def main(args=None):
    rclpy.init(args=args)
    node = LVLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

