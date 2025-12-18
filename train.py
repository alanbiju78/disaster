from ultralytics import YOLO

def main():
    # Load model
    model = YOLO("yolov8n.pt")  # nano version (fast, good for learning)

# Train on your dataset
    model.train(
        data="data.yaml",
        epochs=1,
        imgsz=640,
        batch=16,
        device=0  # use GPU if available
    )

    # Run inference
    # results = model(
    #     source="test.jpg",  # image / video / 0 for webcam
    #     save=True,
    #     conf=0.4
    # )

    print("Inference completed")
if __name__ == "__main__":
    main()
