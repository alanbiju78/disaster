from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")

# Run inference
results = model(
    source="test.jpg",  # image / video / 0 for webcam
    save=True,
    conf=0.4
)

print("Inference completed")
