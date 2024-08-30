from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

model.train(data='config.yaml', epochs=1, imgsz=640)

