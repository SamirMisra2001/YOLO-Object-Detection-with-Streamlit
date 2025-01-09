import streamlit as st
import random
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

# Function to read class list
def read_classes(class_file):
    return class_file.read().decode("utf-8").split("\n")

# Function to generate random colors for detection classes
def generate_colors(class_list):
    return [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in class_list
    ]

# Streamlit app starts here
def main():
    st.title("YOLO Object Detection with Streamlit")
    st.sidebar.header("Configuration")
    st.sidebar.info("Upload model, class file, and video for detection")

    # Upload COCO class file
    class_file = st.sidebar.file_uploader("Upload COCO class file (.txt)", type=["txt"])
    if class_file:
        class_list = read_classes(class_file)
    else:
        st.sidebar.warning("Please upload a COCO class file to continue.")
        return

    # Upload YOLO model file
    model_file = st.sidebar.file_uploader("Upload YOLO model file (.pt)", type=["pt"])
    if model_file:
        with open("temp_model.pt", "wb") as f:
            f.write(model_file.read())
        model = YOLO("temp_model.pt", "v8")
    else:
        st.sidebar.warning("Please upload a YOLO model file to continue.")
        return

    # Confidence threshold
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)

    # Generate colors for detection classes
    detection_colors = generate_colors(class_list)

    # Upload video file
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if video_file:
        # Use BytesIO to handle video
        tfile = BytesIO(video_file.read())
        cap = cv2.VideoCapture(tfile)

        stframe = st.empty()

        with st.spinner("Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform object detection
                results = model.predict(source=[frame], conf=confidence_threshold)
                detections = results[0]

                # Draw detections on the frame
                if len(detections.boxes) != 0:
                    for box in detections.boxes:
                        bb = box.xyxy.numpy()[0]  # Bounding box coordinates
                        clsID = int(box.cls.numpy()[0])  # Class ID
                        conf = box.conf.numpy()[0]  # Confidence score

                        # Draw bounding box
                        cv2.rectangle(
                            frame,
                            (int(bb[0]), int(bb[1])),
                            (int(bb[2]), int(bb[3])),
                            detection_colors[clsID],
                            2,
                        )

                        # Display class name and confidence
                        cv2.putText(
                            frame,
                            f"{class_list[clsID]} {round(conf * 100, 2)}%",
                            (int(bb[0]), int(bb[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

                # Convert the frame to RGB for Streamlit
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()

        st.success("Video processing completed!")

    else:
        st.warning("Please upload a video file to proceed.")

if __name__ == "__main__":
    main()
