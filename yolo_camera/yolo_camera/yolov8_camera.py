import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO


model = YOLO('yolov8n.pt')  # Smallest network from yolov8


class Yolov8Node(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10)

        self.bridge = CvBridge()

    def camera_callback(self, image: Image):
        self.get_logger().info('Connection to camera frame')

        # convert ros2 -> opencv (for yolov8 analysis)
        frame = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        image_frame = frame

        result = model.predict(image_frame, classes=[0])  # classes=[0] is the class list from COCO -> person
        live_img = result[0].plot()


        # show results
        cv2.imshow('Detect', live_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    image_sub = Yolov8Node()

    rclpy.spin(image_sub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
