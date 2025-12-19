from ultralytics import YOLO  # type: ignore

def main():
    model = YOLO("runs/detect/train2/weights/best.pt")

    model(
        source="collage.jpg",   # image / folder / video / 0 for webcam
        conf=0.4,
        save=True,
        device=0
    )

    print("Inference completed")

if __name__ == "__main__":
    main()
