import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from ultralytics import YOLO
import ffmpeg
from pathlib import Path
import time

# --- Streamlit page setup ---
st.set_page_config(page_title="Vehicle Detection Dashboard", layout="wide")
st.title("🚦 Vehicle Detection & Processing Dashboard")
st.markdown("Upload a video, pick a YOLOv8 model, toggle vehicle types, and download the processed output.")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Choose YOLOv8 model size", options=["yolov8n", "yolov8s", "yolov8m", "yolov8l"], index=0)
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.4)
    frame_skip = st.slider("Process every Nth frame (speed/quality)", 1, 10, 1)
    use_gpu = st.checkbox("Use GPU if available", value=True)
    process_button = st.button("Start processing")

# --- COCO class indexes for common vehicles ---
COCO_VEHICLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    4: "aeroplane",
    5: "bus",
    6: "train",
    7: "truck",
}

# --- Vehicle class selection ---
selected_classes = st.multiselect(
    "Select classes to detect",
    options=[f"{k}: {v}" for k, v in COCO_VEHICLE_CLASSES.items()],
    default=["2: car", "3: motorbike", "5: bus", "7: truck"]
)
selected_class_ids = [int(s.split(":")[0]) for s in selected_classes]

# --- File uploader ---
uploaded_file = st.file_uploader("Upload a video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

# --- Video processing ---
if process_button:
    if uploaded_file is None:
        st.warning("Please upload a video file first.")
    elif len(selected_class_ids) == 0:
        st.warning("Select at least one class to detect.")
    else:
        # Save uploaded file to a temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
        tfile.write(uploaded_file.read())
        tfile.flush()
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Could not open uploaded video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Prepare output
            out_path_avi = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(out_path_avi, fourcc, fps, (width, height))

            # Load YOLOv8 model
            model_file = f"{model_choice}.pt"
            st.info(f"Loading model {model_choice}... (this may take a while)")
            try:
                model = YOLO(model_file)
                if use_gpu:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            model.to('cuda')
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Error loading model: {e}")
                cap.release()
                out.release()
                raise

            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            processed_frames = 0
            frame_idx = 0

            # --- Main loop ---
            pbar_text = st.empty()
            start_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    out.write(frame)
                    continue

                # Run YOLO inference
                results = model(frame, conf=conf_threshold, verbose=False)

                # Draw detections
                for res in results:
                    boxes = res.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls not in selected_class_ids:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        label = f"{COCO_VEHICLE_CLASSES.get(cls, str(cls))}: {conf:.2f}"
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                out.write(frame)

                processed_frames += 1
                progress = int(processed_frames / (total_frames / frame_skip) * 100)
                progress_bar.progress(min(progress, 100))
                pbar_text.text(f"Processed frame {frame_idx}/{total_frames}")

            cap.release()
            out.release()

            # Convert to mp4 using ffmpeg
            out_path_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            try:
                ffmpeg.input(out_path_avi).output(out_path_mp4, vcodec='libx264', crf=23).run(overwrite_output=True)
            except Exception as e:
                st.warning(f"ffmpeg conversion failed; serving AVI directly. Error: {e}")
                out_path_mp4 = out_path_avi

            # Show video
            st.success("Processing complete!")
            st.video(out_path_mp4)

            # Download button
            with open(out_path_mp4, "rb") as f:
                data = f.read()
                st.download_button("Download processed video", data, file_name="processed_video.mp4")

            # Clean temp files
            try:
                os.remove(tfile.name)
                if out_path_avi != out_path_mp4:
                    os.remove(out_path_avi)
            except Exception:
                pass

            end_time = time.time()
            st.write(f"Elapsed time: {end_time - start_time:.1f}s")
