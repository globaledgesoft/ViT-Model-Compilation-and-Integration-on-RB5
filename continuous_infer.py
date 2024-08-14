import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import cv2
import json
import sys
from flask import Flask, Response

app = Flask(__name__)

def load_labels(path):
    """Loads the labels file from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def preprocess_image(image_path, target_size):
    """Preprocess the input image to the required input size."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_np = np.array(image).astype(np.float32)
    image_np = image_np / 255.0  # normalize to [0,1]
    return np.expand_dims(image_np, axis=0)

def preprocess_image1(image_frame, target_size):
    """Preprocess the input image frame to the required input size."""
    image = Image.fromarray(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))
    image = image.resize(target_size)
    image_np = np.array(image).astype(np.float32)
    image_np = image_np / 255.0  # normalize to [0,1]
    return np.expand_dims(image_np, axis=0)


def run_inference(interpreter, input_details, output_details, input_data):
    """Run inference on the model."""
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output['index']) for output in output_details]
    return output_data

def print_top_k_predictions(predictions, labels, k=5):
    """Print the top-k predictions."""
    top_k_indices = np.argsort(predictions[0])[-k:][::-1]
    for i in top_k_indices:
        print(f"{labels[i]}: {predictions[0][i]:.4f}")

def capture_and_save_image(camera_index=0, output_path='captured_image.jpg', target_size=(224, 224)):
    """Capture an image from the USB camera and save it."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return False
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from the camera.")
        return False
    resized_frame = cv2.resize(frame, target_size)
    cv2.imwrite(output_path, resized_frame)
    print(f"Image saved at {output_path}")
    cap.release()
    return True

def capture_and_infer(camera_index='/dev/video2', target_size=(224, 224)):
    """Capture image from camera, run inference, and send to endpoint."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    # Path to the TFLite model and labels file
    model_path = './model/vit.tflite'
    labels_path = './labels.json'

    # Load the TFLite model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load labels
    labels = load_labels(labels_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the camera.")
            break

        # Preprocess the input image
        input_data = preprocess_image1(frame, target_size)

        # Run inference
        output_data = run_inference(interpreter, input_details, output_details, input_data)

        # Get the top prediction
        predictions = output_data[0][0]
        top_prediction_index = np.argmax(predictions)
        predicted_label = labels[top_prediction_index]
        confidence_score = predictions[top_prediction_index]

        # Add predicted class label and confidence score to the image
        text = f"Predicted Class: {predicted_label} ({confidence_score:.4f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 20

        # Draw a black rectangle behind the text
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        # Put the text on top of the black rectangle
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes() 

        # Yield the frame to the endpoint
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def infer_on_image(use_camera=False):
    # Path to the TFLite model and labels file
    model_path = './model/vit.tflite'
    labels_path = './labels.json'

    # Load the TFLite model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load labels
    labels = load_labels(labels_path)

    # Image path setup
    image_path = './images/jaguar.jpg'  # Default image path
    if use_camera:
        camera_index = '/dev/video2'  # Adjust this based on your camera index
        success = capture_and_save_image(camera_index=camera_index, output_path='captured_image.jpg')
        if not success:
            return
        image_path = './captured_image.jpg'

    # Preprocess the input image
    input_size = (224, 224)
    input_data = preprocess_image(image_path, input_size)

    # Run inference
    output_data = run_inference(interpreter, input_details, output_details, input_data)

    # Print output data for debugging
    print("Output data shapes:")
    for i, output in enumerate(output_data):
        print(f"Output {i}: shape {output.shape}")

    # Print the top-5 predictions
    print_top_k_predictions(output_data[0], labels)

    # Save the image with the highest classification result
    predictions = output_data[0][0]
    top_prediction_index = np.argmax(predictions)
    predicted_label = labels[top_prediction_index]
    confidence_score = predictions[top_prediction_index]

    print(f"Predicted Class: {predicted_label} with confidence: {confidence_score:.4f}")

    # Load the original image using OpenCV
    image = cv2.imread(image_path)

    # Add predicted class label and confidence score to the image
    text = f"Predicted Class: {predicted_label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = image.shape[0] - 20

    # Draw a black rectangle behind the text
    cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    # Put the text on top of the black rectangle
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Save the image with the predicted class label
    output_image_path = './classified_image.jpg'
    cv2.imwrite(output_image_path, image)
    print(f"Image saved at {output_image_path}")

def infer_on_jaguar(image_path, target_size=(224, 224)):
    """Run inference on a single image and print the result."""
    # Path to the TFLite model and labels file
    model_path = './model/vit.tflite'
    labels_path = './labels.json'

    # Load the TFLite model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load labels
    labels = load_labels(labels_path)

    # Preprocess the input image
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_np = np.array(image).astype(np.float32)
    image_np = image_np / 255.0  # normalize to [0,1]
    input_data = np.expand_dims(image_np, axis=0)

    # Run inference
    output_data = run_inference(interpreter, input_details, output_details, input_data)

    # Get the top prediction
    predictions = output_data[0][0]
    top_prediction_index = np.argmax(predictions)
    predicted_label = labels[top_prediction_index]
    confidence_score = predictions[top_prediction_index]

    print(f"Predicted Class: {predicted_label} ({confidence_score:.4f})")
    image = cv2.imread(image_path)

    # Add predicted class label and confidence score to the image
    text = f"Predicted Class: {predicted_label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = image.shape[0] - 20

    # Draw a black rectangle behind the text
    cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    # Put the text on top of the black rectangle
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Save the image with the predicted class label
    output_image_path = './classified_image.jpg'
    cv2.imwrite(output_image_path, image)
    print(f"Image saved at {output_image_path}")


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(capture_and_infer(camera_index='/dev/video2'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--feed':
            app.run(host='0.0.0.0', port=5000)
        elif sys.argv[1] == '--camera':
            use_camera = len(sys.argv) > 1 and sys.argv[1] == '--camera'
            infer_on_image(use_camera=use_camera)
    else:
        infer_on_jaguar('./jaguar.jpg')
