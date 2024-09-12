import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

# Initialize YOLO model
model = YOLO("lol.pt")

# Streamlit app title
st.title("YOLO Object Tracking")

# Sidebar for confidence and IoU settings
with st.sidebar:
    st.title("Tracking Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3)
    iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.13)
    
    # Dropdown to choose between bytetrack.yaml and botsort.yaml
    tracker_config = st.selectbox("Select Tracker Configuration", ("bytetrack.yaml", "botsort.yaml"))

    # Radio button to select tracking mode (real-time or video)
    tracking_mode = st.radio("Select Tracking Mode", ("Real-Time (Webcam)", "Upload Video"))

# Centered video output and larger display
col_video = st.columns([0.5, 8, 0.5])

# Streamlit session state to store video capture object and other states
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'video_source' not in st.session_state:
    st.session_state.video_source = ""
if 'start_tracking' not in st.session_state:
    st.session_state.start_tracking = False
if 'last_tracking_mode' not in st.session_state:
    st.session_state.last_tracking_mode = ""

def reset_session_state():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    st.session_state.cap = None
    st.session_state.start_tracking = False

def update_tracking_state():
    reset_session_state()

# Check if tracking mode has changed
if tracking_mode != st.session_state.last_tracking_mode:
    update_tracking_state()
    st.session_state.last_tracking_mode = tracking_mode

# Initialize tracking history
track_history = {}

def update_track_history(track_id, box_center):
    if track_id not in track_history:
        track_history[track_id] = []
    track_history[track_id].append(box_center)
    if len(track_history[track_id]) > 30:  # retain 30 tracks for 30 frames
        track_history[track_id].pop(0)

# Real-time processing
if tracking_mode == "Real-Time (Webcam)":
    st.write("Real-Time Tracking Mode Selected")
    
    # Button to start webcam capture and YOLO tracking
    start_tracking = st.button("Start Real-Time Tracking")
    
    if start_tracking:
        st.session_state.start_tracking = True
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.video_source = "webcam"
    
    if st.session_state.start_tracking and st.session_state.cap is not None:
        frame_skip = 2
        frame_count = 0
        
        with col_video[1]:
            stframe = st.empty()  # Placeholder for real-time video frame display

            while st.session_state.cap.isOpened():
                ret, frame = st.session_state.cap.read()
                frame_count += 1

                if not ret or frame_count % frame_skip != 0:
                    continue  # Skip frames
                
                # Resize for faster processing
                frame = cv2.resize(frame, (640, 480))

                # Run the YOLO model on the frame
                results = model.track(frame, conf=confidence_threshold, iou=iou_threshold, tracker=tracker_config, persist=True)

                boxes = results[0].boxes.xywh.cpu()
                track_ids = (
                    results[0].boxes.id.int().cpu().tolist()
                    if results[0].boxes.id is not None
                    else None
                )

                annotated_frame = results[0].plot()

                # Plot the tracks
                if track_ids:
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        box_center = (float(x) + w / 2, float(y) + h / 2)
                        update_track_history(track_id, box_center)

                        if track_id in track_history:
                            points = np.array(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                            # Remove the following line to avoid drawing white lines
                            # cv2.polylines(
                            #     annotated_frame,
                            #     [points],
                            #     isClosed=False,
                            #     color=(230, 230, 230),
                            #     thickness=2,
                            # )
                        
                # Convert frame from BGR (OpenCV format) to RGB for display in Streamlit
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display the frame in the Streamlit app
                stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        st.session_state.cap.release()
        st.session_state.start_tracking = False

# Upload video processing
elif tracking_mode == "Upload Video":
    st.write("Upload a Video for YOLO Object Tracking")
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        
        st.session_state.cap = cv2.VideoCapture(temp_file.name)
        frame_skip = 2
        frame_count = 0

        if st.session_state.cap.isOpened():
            with col_video[1]:
                stframe = st.empty()  # Placeholder for video frame display

                while st.session_state.cap.isOpened():
                    ret, frame = st.session_state.cap.read()
                    frame_count += 3

                    if not ret or frame_count % frame_skip != 0:
                        continue  # Skip frames
                    
                    frame = cv2.resize(frame, (640, 480))  # Resize the frame for faster processing

                    # Run YOLO on the frame
                    results = model.track(frame, conf=confidence_threshold, iou=iou_threshold, tracker=tracker_config, persist=True)

                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = (
                        results[0].boxes.id.int().cpu().tolist()
                        if results[0].boxes.id is not None
                        else None
                    )

                    annotated_frame = results[0].plot()

                    # Plot the tracks
                    if track_ids:
                        for box, track_id in zip(boxes, track_ids):
                            x, y, w, h = box
                            box_center = (float(x) + w / 2, float(y) + h / 2)
                            update_track_history(track_id, box_center)

                            if track_id in track_history:
                                points = np.array(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                                # Remove the following line to avoid drawing white lines
                                # cv2.polylines(
                                #     annotated_frame,
                                #     [points],
                                #     isClosed=False,
                                #     color=(230, 230, 230),
                                #     thickness=2,
                                # )
                    
                    # Convert frame from BGR to RGB for display
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # Display the frame in the Streamlit app
                    stframe.image(frame_rgb, channels="RGB", use_column_width=True)

            st.session_state.cap.release()
