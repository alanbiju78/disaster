from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # nano version (fast, good for learning)

# Train on your dataset
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
