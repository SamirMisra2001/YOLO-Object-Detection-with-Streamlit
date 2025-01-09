# YOLO-Object-Detection-with-Streamlit
**Overview** </br>
</br>
This project is a web-based application for object detection using the YOLO algorithm. Built with Streamlit, it allows users to upload a YOLO model, class files, and videos to perform object detection in real time.


</br>

**Features** </br>
- Upload and use custom YOLO models for object detection.
- Supports COCO class files for class labels.
- Adjustable confidence thresholds for detections.
- Visualizes bounding boxes and class names on uploaded video files.
- Easy-to-use web interface powered by Streamlit.

</br>

**Dependencies** </br>
- Streamlit 
- OpenCV
- ultralytics
- Pillow
- NumPy

</br>

**How It Works** </br>
- <b>Model Loading:</b> Upload a YOLO model file (.pt) trained using the YOLOv8 framework. 
- <b>Class File:</b> Provide a text file containing class labels (one per line).
- <b>Video Processing:</b> </br>
   I. The uploaded video is processed frame by frame.</br>
   II. Detected objects are annotated with bounding boxes, class names, and confidence scores.
- <b>Results Display:</b> Processed frames are displayed in real-time on the Streamlit interface.

</br>
