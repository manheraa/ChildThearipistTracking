# ChildThearipistTracking

## Project Description

This project demonstrates how to build an object tracking application using YOLO (You Only Look Once) integrated with a Streamlit web app. The application supports real-time object tracking via webcam and video file uploads. The YOLO model, provided by the Ultralytics library, detects and tracks objects in video frames. Users can adjust the confidence and IoU thresholds through the Streamlit interface and choose between different tracking configurations.
## Installation

To install the necessary dependencies for this project, run the following command:

```bash
python3 -m venv venv
For windows:
ven/Scripts/activate
For Linux:
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
  

### Features

- **Real-Time Tracking**: Track objects from a webcam feed with adjustable settings.
- **Video Upload**: Upload video files to perform object tracking on existing footage.
- **Customizable Tracking**: Adjust confidence and IoU thresholds and select different tracker configurations (e.g., `bytetrack.yaml`, `botsort.yaml`).

## Prerequisites

- **Python**: Version 3.12.4
- **CUDA** (optional for GPU acceleration): Version 11.8
- **PyTorch** (optional for GPU acceleration): Version 2.2.2+cu118

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
