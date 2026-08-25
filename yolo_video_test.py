from ultralytics import YOLO

model = YOLO('yolov10s.pt')
results = model.predict(
    r"C:\Users\A8IN\Downloads\WhatsApp Video 2026-08-19 at 4.22.01 PM.mp4",
    save=True,
    conf=0.15,
    imgsz=1280,
    classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
    device=0  # forces your RTX 4050
)
print('Saved to:', results[0].save_dir)