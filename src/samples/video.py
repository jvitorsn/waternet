import os
import cv2
import numpy as np
import pandas as pd

def create_hud_video(image_folder, df, output_filename='output_224.mp4', fps=60, size=(224, 224)):
    """
    Creates a video from a folder of 224x224 images, extending the frame to include HUD data.

    Args:
        image_folder (str): Path to the folder containing images.
        df (pd.DataFrame): Dataframe containing ['filename', 'height', 'altitude', 'lidar'].
        output_filename (str): Path for the output .mp4 file.
        fps (int): Frames per second for the video.
    """
    # Get list of images sorted by name
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print("No images found in the specified folder.")
        return

    # Original dimensions
    orig_h, orig_w = size

    # New dimensions (Adding 176 pixels to the right for the HUD text)
    new_w = orig_w + 224
    new_h = orig_h

    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (new_w, new_h))

    i = 0
    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        # Resize to guarantee 224x224 just in case
        frame = cv2.resize(frame, (orig_w, orig_h))

        # Create a black canvas
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        # Paste the original image onto the left side of the canvas
        canvas[:, :orig_w] = frame
        
        # Fetch data from dataframe based on filename
        row = df[df['filename'] == img_name]
        # HUD Text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        if not row.empty:
            height_val = row.iloc[0]['height']
            altitude_val = row.iloc[0]['altitude']
            lidar_val = row.iloc[0]['lidar']
            waternet_val = row.iloc[0]['waternet']
            waternet_fus_val = row.iloc[0]['waternet_fus']

            color = (0, 255, 0) # Green text
            thickness = 1

            # Add text to the right side (the extended area)

        else:
            pass

        cv2.putText(canvas, "HUD Data", (orig_w + 10, 30), font, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"Height: {height_val}", (orig_w + 10, 50), font, font_scale, color, thickness)
        cv2.putText(canvas, f"Altitude: {altitude_val}", (orig_w + 10, 70), font, font_scale, color, thickness)
        cv2.putText(canvas, f"Lidar: {lidar_val}", (orig_w + 10, 90), font, font_scale, color, thickness)
        cv2.putText(canvas, f"WaterNet: {waternet_val}", (orig_w + 10, 110), font, font_scale, color, thickness)
        cv2.putText(canvas, f"WaterNet_Fus: {waternet_fus_val}", (orig_w + 10, 130), font, font_scale, color, thickness)

        # Write the modified frame to the video
        out.write(canvas)
        i += 1
        if i % 100 == 0:
            print(f"Processed frame {i}/{len(image_files)}: {img_name}")

    out.release()
    print(f"Video saved successfully to {output_filename}")

df = pd.read_csv('./samples/section1/waternet_video.csv', sep=',')
df.head()

img_folder = './samples/section1/section1_224'

create_hud_video(
    image_folder=img_folder,
    df=df
)