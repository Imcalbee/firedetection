from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


# Train the model
results = model.train(data="Some-Test-From-Public-Project-1/data.yaml", epochs=100, imgsz=256)

model = YOLO("./runs/detect/train2/weights/best.pt")

results = model.predict(source="Some-Test-From-Public-Project-1/valid/images")

from PIL import Image
import cv2
from IPython.display import display

for result in results[5:10]:
    img = result.plot()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img)
    display(pil_image)
    
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')

model.export(format='openvino')