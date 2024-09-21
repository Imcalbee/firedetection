import streamlit as st
import openvino as ov
import yaml
import cv2
import numpy as np
from ultralytics.utils.plotting import colors
import time

# Load OpenVINO model
def load_model(model_path, device="AUTO"):
    core = ov.Core()
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    return compiled_model, input_layer, output_layer

# Load metadata
def load_metadata(metadata_path):
    with open(metadata_path) as info:
        info_dict = yaml.load(info, Loader=yaml.Loader)
    labels = info_dict['names']
    return labels

# Preprocessing
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

# Prepare data
def prepare_data(image, input_layer):
    input_w, input_h = input_layer.shape[2], input_layer.shape[3]
    input_image = letterbox(np.array(image))[0]
    input_image = cv2.resize(input_image, (input_w, input_h))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, 0)
    return input_image

# Evaluation function
def evaluate(output, conf_threshold):
    boxes = []
    scores = []
    label_key = []
    for label_index, class_ in enumerate(output[0][4:]):
        for index, confidence in enumerate(class_):
            if confidence > conf_threshold:
                xcen = output[0][0][index]
                ycen = output[0][1][index]
                w = output[0][2][index]
                h = output[0][3][index]
                xmin = int(xcen - (w / 2))
                xmax = int(xcen + (w / 2))
                ymin = int(ycen - (h / 2))
                ymax = int(ycen + (h / 2))
                box = (xmin, ymin, xmax, ymax)
                boxes.append(box)
                scores.append(confidence)
                label_key.append(label_index)
    return np.array(boxes), np.array(scores), label_key

# Main prediction function
def predict_image(image, compiled_model, input_layer, output_layer, conf_threshold=0.4):
    input_image = prepare_data(image, input_layer)
    output = compiled_model([input_image])[output_layer]
    boxes, scores, label_key = evaluate(output, conf_threshold)
    return boxes, scores, label_key

# Visualization
def visualize(image, boxes, label_key, scores, labels):
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        label = label_key[i]
        color = colors(label)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        label_text = f"{int(scores[i] * 100)}% {labels[label]}"
        cv2.putText(image, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Streamlit App
st.title("OpenVINO Object Detection with Streamlit")

# Upload model and image
model_file = st.file_uploader("Upload OpenVINO model (.xml)", type=["xml"])
metadata_file = st.file_uploader("Upload Metadata (.yaml)", type=["yaml"])
image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if model_file and metadata_file and image_file:
    with st.spinner("Loading model and metadata..."):
        compiled_model, input_layer, output_layer = load_model(model_file)
        labels = load_metadata(metadata_file)

    image = np.array(cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1))

    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4)

    # Predict and visualize
    boxes, scores, label_key = predict_image(image, compiled_model, input_layer, output_layer, conf_threshold)
    output_image = visualize(image, boxes, label_key, scores, labels)

    st.image(output_image, caption="Processed Image", use_column_width=True)
