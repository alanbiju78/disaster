from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")

model(
    source="image.jpg",
    conf=0.4,
    save=True,
    device=0
)
