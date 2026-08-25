from ultralytics import YOLO

model = YOLO('yolov10s.pt')  # 's' (small) instead of 'n' (nano) — better accuracy, still fast
results = model.predict(
    r'C:\Users\A8IN\Downloads\archive\content\UA-DETRAC\DETRAC_Upload\images\val\MVI_39031_img00001.jpg',
    save=True,
    conf=0.15,
    imgsz=1280,
    classes=[2, 3, 5, 7]  # COCO IDs: 2=car, 3=motorcycle, 5=bus, 7=truck — restricts output to vehicles only
)
print('Saved to:', results[0].save_dir)