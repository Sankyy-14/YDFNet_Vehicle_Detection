from ultralytics import YOLO

model = YOLO('yolov8s-world.pt')
model.set_classes(['auto rickshaw', 'tuk-tuk', 'car', 'bus', 'truck'])

results = model.predict(
    r"C:\Users\A8IN\Downloads\WhatsApp Video 2026-08-21 at 12.19.28 AM.mp4",
    save=True,
    conf=0.15,
    device=0
)
print('Saved to:', results[0].save_dir)