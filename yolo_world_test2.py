from ultralytics import YOLO

model = YOLO('yolov8l-world.pt')  # larger model, stronger text-matching
model.set_classes(['three-wheeler auto rickshaw', 'tuk-tuk taxi'])

results = model.predict(
    r"C:\Users\A8IN\Downloads\WhatsApp Video 2026-08-21 at 12.19.28 AM.mp4",
    save=True,
    conf=0.1,
    device=0
)
print('Saved to:', results[0].save_dir)