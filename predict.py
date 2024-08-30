from ultralytics import YOLO

model = YOLO('runs/segment/train2/weights/best.pt')

result = model('windows.jpg')

