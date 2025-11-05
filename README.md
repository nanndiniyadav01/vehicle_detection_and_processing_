# 🚦 Vehicle Detection & Processing Dashboard

A **Streamlit app** that detects vehicles in videos using **YOLOv8** and allows you to visualize, process, and download the results. This project is a practical introduction to **computer vision, object detection, and video processing**.

---

## 🔹 Project Overview

Modern traffic systems and urban planning can greatly benefit from automated vehicle detection. This app allows users to:

- Upload a video (MP4, AVI, MOV) of traffic or vehicles.
- Choose a **YOLOv8 model** size (`n`, `s`, `m`, `l`) for detection.
- Select specific **vehicle classes** to detect: car, bus, truck, motorbike.
- Visualize bounding boxes with confidence scores on detected vehicles.
- Download the **processed video** for further analysis.

This is ideal for **learning computer vision**, prototyping traffic monitoring systems, or exploring YOLOv8 for video analysis.

---

## 🚀 Features

- **Multi-class vehicle detection** using YOLOv8.
- **Interactive Streamlit interface** for easy uploads and processing.
- **Frame-wise processing** with adjustable frame skipping for speed/accuracy balance.
- **Bounding box visualization** with class labels.
- **Download processed video** for offline use.
- **Optional centroid-based tracking** for simple vehicle tracking (can be enhanced further).

---

## 🎥 Sample Video

A small sample video (`video.mp4`) is included for quick testing.

---

## ⚙️ Getting Started

### **1. Clone the repository**
```bash
git clone https://github.com/nanndiniyadav01/vehicle_detection_and_processing_.git
cd vehicle_detection_and_processing_
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit app**
```bash
streamlit run vehicle_detection_streamlit.py
```

---

## 🛠 File Structure

```
vehicle_detection_streamlit.py   # Main Streamlit app
video.mp4                        # Sample video for testing
requirements.txt                 # Python dependencies
README.md                        # Project documentation
```

---

## 🔮 Future Advancements

This project is highly extendable. Possible improvements include:

1. **Real-time detection** using webcam or live video streams.
2. **Advanced vehicle tracking** with SORT or DeepSORT for tracking IDs across frames.
3. **Traffic analytics**:
   - Count vehicles per type.
   - Detect congestion patterns.
   - Calculate average speed if timestamps are available.
4. **Alert system** for unusual events (e.g., wrong-way vehicles, stopped vehicles).
5. **Web dashboard** integration with charts for vehicle counts and trends.
6. **AI-powered insights** like predicting traffic flow or accident-prone areas.
7. **Integration with GIS maps** to visualize vehicle movement on real roads.

---

## 🧰 Tech Stack

- **Python** – programming language
- **Streamlit** – for interactive web interface
- **OpenCV** – video reading and processing
- **YOLOv8 (Ultralytics)** – object detection
- **NumPy** – numerical computations
- **ffmpeg-python** – video encoding/decoding
- **PyTorch** – deep learning backend

---

## 💡 Notes

- Best tested with small/medium videos (<50 MB).  
- Model selection affects **accuracy vs. speed**:  
  - `yolov8n` → fastest, less accurate  
  - `yolov8l` → slower, most accurate

---

## 📄 License

MIT License – feel free to use, modify, and extend this project.

---

## ⭐ Contributing

This project is open for contributions! Some ideas:

- Add more vehicle types (e.g., bicycles, trucks, buses in different sizes).  
- Integrate with a **real-time traffic camera feed**.  
- Improve tracking and counting logic for accuracy.
