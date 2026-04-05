from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

BAG_PATH    = "./rosbags/section1.bag"
IMAGE_TOPIC = "/pylon_camera_node/image_raw/compressed"
HEIGHT_TOPIC  = "/uav62/odometry/height"
ALTITUDE_TOPIC = "/uav62/odometry/altitude"
LIDAR_TOPIC   = "/uav62/mavros/distance_sensor/garmin"
TOPICS_NEEDED = [
    IMAGE_TOPIC,
    HEIGHT_TOPIC,
    ALTITUDE_TOPIC,
    LIDAR_TOPIC
    ]

OUTPUT_DIR  = Path("extracted_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

typestore = get_typestore(Stores.ROS1_NOETIC)
with Reader(BAG_PATH) as reader:
    add_types = {}
    for conn in reader.connections:
        if conn.topic not in TOPICS_NEEDED:
            continue
        msgdef = conn.msgdef.data if hasattr(conn.msgdef, 'data') else conn.msgdef
        try:
            add_types.update(get_types_from_msg(msgdef, conn.msgtype))
        except Exception:
            pass
    typestore.register(add_types)

height_records = []   # (timestamp_ns, z)
altitude_records = [] # (timestamp_ns, z)
lidar_records = []    # (timestamp_ns, z)
image_records = []  # (timestamp_ns, filename)

with Reader(BAG_PATH) as reader:
    img_conns  = [c for c in reader.connections if c.topic == IMAGE_TOPIC]
    height_conns = [c for c in reader.connections if c.topic == HEIGHT_TOPIC]
    altitude_conns = [c for c in reader.connections if c.topic == ALTITUDE_TOPIC]
    lidar_conns = [c for c in reader.connections if c.topic == LIDAR_TOPIC]
    all_conns  = img_conns + height_conns + altitude_conns + lidar_conns

    for i, (conn, timestamp, rawdata) in enumerate(reader.messages(connections=all_conns)):
        msg = typestore.deserialize_ros1(rawdata, conn.msgtype)

        if conn.topic == IMAGE_TOPIC:
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                continue
            fname = f"frame_{i:06d}.jpg"
            # cv2.imwrite(str(OUTPUT_DIR / fname), img)
            image_records.append({"timestamp_ns": timestamp, "filename": fname})

        elif conn.topic == HEIGHT_TOPIC:
            # nav_msgs/Odometry — adjust field path for other message types
            height_records.append({"timestamp_ns": timestamp, "HEIGHT": msg.value})

        elif conn.topic == ALTITUDE_TOPIC:
            # nav_msgs/Odometry — adjust field path for other message types
            altitude_records.append({"timestamp_ns": timestamp, "ALTITUDE": msg.value})

        elif conn.topic == LIDAR_TOPIC:
            # sensor_msgs/Range — adjust field path for other message types
            lidar_records.append({"timestamp_ns": timestamp, "LIDAR": msg.range})

print(f"Finished reading bag. Total messages: {i+1}")
img_df  = pd.DataFrame(image_records)   # timestamp_ns, filename
height_df = pd.DataFrame(height_records)    # timestamp_ns, HEIGHT
altitude_df = pd.DataFrame(altitude_records)  # timestamp_ns, ALTITUDE
lidar_df = pd.DataFrame(lidar_records)      # timestamp_ns, LIDAR
img_df.to_csv("image_records.csv", index=False)
height_df.to_csv("height_records.csv", index=False)
altitude_df.to_csv("altitude_records.csv", index=False)
lidar_df.to_csv("lidar_records.csv", index=False)

# Sort both by time
img_df  = img_df.sort_values("timestamp_ns").reset_index(drop=True)
height_df = height_df.sort_values("timestamp_ns").reset_index(drop=True)
altitude_df = altitude_df.sort_values("timestamp_ns").reset_index(drop=True)
lidar_df = lidar_df.sort_values("timestamp_ns").reset_index(drop=True)  

# For each image timestamp, find the closest height timestamp
height_ts = height_df["timestamp_ns"].values
altitude_ts = altitude_df["timestamp_ns"].values
lidar_ts = lidar_df["timestamp_ns"].values

def find_nearest_height(img_ts):
    idx = np.searchsorted(height_ts, img_ts)
    idx = np.clip(idx, 0, len(height_ts) - 1)
    # Check neighbour on both sides
    if idx > 0 and abs(height_ts[idx-1] - img_ts) < abs(height_ts[idx] - img_ts):
        idx -= 1
    return height_df.loc[idx, "HEIGHT"]

def find_nearest_altitude(img_ts):
    idx = np.searchsorted(altitude_ts, img_ts)
    idx = np.clip(idx, 0, len(altitude_ts) - 1)
    if idx > 0 and abs(altitude_ts[idx-1] - img_ts) < abs(altitude_ts[idx] - img_ts):
        idx -= 1
    return altitude_df.loc[idx, "ALTITUDE"]

def find_nearest_lidar(img_ts):
    idx = np.searchsorted(lidar_ts, img_ts)
    idx = np.clip(idx, 0, len(lidar_ts) - 1)
    if idx > 0 and abs(lidar_ts[idx-1] - img_ts) < abs(lidar_ts[idx] - img_ts):
        idx -= 1
    return lidar_df.loc[idx, "LIDAR"]

img_df["height"] = img_df["timestamp_ns"].apply(find_nearest_height)
img_df["altitude"] = img_df["timestamp_ns"].apply(find_nearest_altitude)
img_df["lidar"] = img_df["timestamp_ns"].apply(find_nearest_lidar)

# Convert ns → seconds for readability
img_df["timestamp_s"] = img_df["timestamp_ns"] / 1e9

img_df.to_csv("image_height_positions.csv", index=False) # [["filename", "timestamp_s", "height", "altitude", "lidar"]]
print(img_df.head())
