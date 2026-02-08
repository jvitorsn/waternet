#!/usr/bin/env python3
"""Extract sensor data and video frames from ROS bag files.

This script reads a ROS bag file and extracts:
- Sensor readings to CSV files
- Video frames to PNG images
"""

from pathlib import Path
from typing import Dict, List, Any
import csv

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores
typestore = get_typestore(Stores.ROS1)

from rosbags.rosbag1 import Reader
# from rosbags.serde import deserialize_cdr # This line is incorrect

def extract_sensor_data(
    bag_path: Path,
    sensor_topics: List[str],
    output_dir: Path,
) -> dict:
    """Extract sensor readings from specified topics to CSV files.
    
    Args:
        bag_path: Path to the ROS bag file.
        sensor_topics: List of topic names to extract.
        output_dir: Directory to save CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data buffers for each topic
    data_buffers: Dict[str, List[Dict[str, Any]]] = {
        topic: [] for topic in sensor_topics
    }
    
    with Reader(bag_path) as reader:
        # Filter connections by topic
        connections = [
            conn for conn in reader.connections
            if conn.topic in sensor_topics
        ]
        
        # Read messages
        for connection, timestamp, rawdata in reader.messages(
            connections=connections
        ):
            msg = deserialize_cdr(rawdata, connection.msgtype)
            
            # Extract fields recursively
            record = {'timestamp': timestamp}
            record.update(_flatten_message(msg))
            data_buffers[connection.topic].append(record)
    
    return data_buffers
    
def writeCSV(
    data_buffers: dict, 
    output_dir: str
    ) -> None:
    # Write CSV files
    for topic, records in data_buffers.items():
        if not records:
            continue
            
        csv_path = output_dir / f"{topic.replace('/', '_')}.csv"
        fieldnames = records[0].keys()
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def extract_video_frames(
    bag_path: Path,
    image_topic: str,
    output_dir: Path,
    frame_skip: int = 1,
) -> None:
    """Extract video frames from image topic to PNG files.
    
    Args:
        bag_path: Path to the ROS bag file.
        image_topic: Topic name containing image messages.
        output_dir: Directory to save PNG frames.
        frame_skip: Extract every Nth frame (1 = all frames).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    
    with Reader(bag_path) as reader:
        connections = [
            conn for conn in reader.connections
            if conn.topic == image_topic
        ]
        
        for connection, timestamp, rawdata in reader.messages(
            connections=connections
        ):
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
                
            msg = deserialize_cdr(rawdata, connection.msgtype)
            
            # Convert ROS image to numpy array
            if msg.encoding == 'rgb8':
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
            elif msg.encoding == 'mono8':
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width
                )
            else:
                print(f"Unsupported encoding: {msg.encoding}")
                frame_count += 1
                continue
            
            # Save frame
            frame_path = output_dir / f"frame_{frame_count:06d}.png"
            cv2.imwrite(str(frame_path), image)
            frame_count += 1


def _flatten_message(msg: Any, prefix: str = '') -> Dict[str, Any]:
    """Recursively flatten ROS message to dictionary.
    
    Args:
        msg: ROS message object.
        prefix: Key prefix for nested fields.
        
    Returns:
        Flattened dictionary of message fields.
    """
    result = {}
    
    for field_name in dir(msg):
        if field_name.startswith('_'):
            continue
            
        value = getattr(msg, field_name, None)
        key = f"{prefix}{field_name}" if prefix else field_name
        
        # Handle nested messages
        if hasattr(value, '__dict__') and not isinstance(value, (str, bytes)):
            result.update(_flatten_message(value, f"{key}."))
        # Handle arrays/lists
        elif isinstance(value, (list, tuple)) and value:
            if hasattr(value[0], '__dict__'):
                # Skip complex nested arrays
                continue
            else:
                result[key] = str(value)
        else:
            result[key] = value
            
    return result

def readMsg(bagpath):
    with Reader(bagpath) as reader:
        # Ensure custom message types are registered if necessary (not always needed for core types)
        # reader.connections contain message definitions which can be registered if needed.
        for connection, timestamp, rawdata in reader.messages():
            # Use the typestore to deserialize the raw data
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype) 
            # For ROS2 bags, you would use typestore.deserialize_cdr()

            print(msg)


if __name__ == '__main__':
    # Configuration
    BAG_PATH = Path('./samples/_2022-04-27-16-10-55-002.bag')
    OUTPUT_DIR = Path('extracted_data')
    
    readMsg(BAG_PATH)

    # Extract sensor data
    SENSOR_TOPICS = [
        '/imu/data',
        '/gps/fix',
        '/odom',
    ]
    sensor_data = extract_sensor_data(BAG_PATH, SENSOR_TOPICS, OUTPUT_DIR / 'sensors')
    print(sensor_data.keys())

    if None:
        # Extract video frames
        IMAGE_TOPIC = '/camera/image_raw'
        extract_video_frames(
            BAG_PATH,
            IMAGE_TOPIC,
            OUTPUT_DIR / 'frames',
            frame_skip=5,  # Extract every 5th frame
        )
    
    print(f"Extraction complete. Data saved to {OUTPUT_DIR}")