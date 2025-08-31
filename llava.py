import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
from rclpy.duration import Duration
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

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

        # Load LLaVA model once
        self.get_logger().info("Loading LLaVA model...")
        model_id = "llava-hf/llava-1.5-7b-hf"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        self.model.eval()
        self.get_logger().info("LLaVA model loaded!")

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
            prompt = (
                f"USER: <image>\n"
                f"Find {self._target_object} in this scene. "
                f"List three unique objects or areas that are most helpful as clues or context to locate the {self._target_object}. "
                f"Do NOT include the {self._target_object} itself, its name, or variations of its name. "
                f"Write ONLY the object or area names as a plain comma-separated list in lowercase. "
                f"Do not use 'a', 'an', or 'the'. Do not add any extra words.\n"
                "ASSISTANT:")

            print("prompt: ", prompt)

            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Preprocess and generate
            inputs = self.processor(text=prompt, images=cv_image, return_tensors="pt").to("cuda")
            for k,v in inputs.items():
                print(k,v.shape)
            output = self.model.generate(**inputs, max_new_tokens=20)
            generated_text = self.processor.batch_decode(output, skip_special_tokens=True)

            #Extract only the part after ASSISTANT:
            objects_text = generated_text[0].split("ASSISTANT:")[-1].strip()
            print("output_text: ", objects_text)

            # Publish
            self.output_pub.publish(String(data=objects_text))

            self.last_call = now
            self.get_logger().info("Published LVLM output")

            self.trigger_active = False
        else:
            return

def main(args=None):
    rclpy.init(args=args)
    node = LVLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

