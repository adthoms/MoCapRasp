#! /usr/bin/env python
import os
import cv2
import glob
import rospy
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

def convert_images_to_bag(image_folder, bag_file, topic):
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder {image_folder} does not exist.")
    
    bag = rosbag.Bag(bag_file, 'w')
    bridge = CvBridge()

    images = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
    t = rospy.Time.now()

    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            rospy.logwarn(f"Could not read image {img_path}. Skipping.")
            continue
        
        # Convert OpenCV image to ROS Image message
        img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header = Header()
        img_msg.header.stamp = t + rospy.Duration(i * 0.1)
        bag.write(topic, img_msg, img_msg.header.stamp)
    
    bag.close()
    rospy.loginfo(f"Converted {len(images)} images to {bag_file} on topic {topic}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert images to ROS bag format.")
    parser.add_argument('--input', type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output ROS bag file.")
    parser.add_argument('--topic', type=str, default="/cam1/image_raw", help="ROS topic for the images.")
    args = parser.parse_args()

    rospy.init_node('image_to_bag_writer', anonymous=True)
    
    convert_images_to_bag(args.input, args.output, args.topic)