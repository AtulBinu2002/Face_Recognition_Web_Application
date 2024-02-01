from flask import Flask, render_template, request
from deepface import DeepFace
import cv2
import numpy as np
import base64
import os
import face_recognition
import uuid

app = Flask(__name__)

DATABASE_PATH = 'static/images/'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def is_deep_fake(image_path):
#     # Load the image file
#     image = face_recognition.load_image_file(image_path)

#     # Find all face locations in the image
#     face_locations = face_recognition.face_locations(image)

#     if not face_locations:
#         print("No faces found in the image.")
#         return False

#     # Check for manipulation by comparing the first face with the entire image
#     first_face_encoding = face_recognition.face_encodings(image, known_face_locations=[face_locations[0]])[0]
#     full_image_encoding = face_recognition.face_encodings(image)[0]

#     # Compare the encodings
#     results = face_recognition.compare_faces([first_face_encoding], full_image_encoding, tolerance=0.6)

#     # Calculate confidence for deep fake detection
#     confidence = 1.0 - face_recognition.face_distance([first_face_encoding], full_image_encoding)[0]

#     return not any(results), confidence


def load_face_encodings():
    face_encodings = {}
    
    for person_name in os.listdir(DATABASE_PATH):
        person_folder = os.path.join(DATABASE_PATH, person_name)

        if os.path.isdir(person_folder):
            image_files = [f for f in os.listdir(person_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

            if image_files:
                person_encodings = []

                for image_file in image_files:
                    image_path = os.path.join(person_folder, image_file)
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)[0]
                    person_encodings.append(encoding)

                face_encodings[person_name] = person_encodings

    return face_encodings


def recognize_person(cropped_face_base64, face_encodings):
    # Decode base64 and convert to numpy array
    cropped_face_buffer = base64.b64decode(cropped_face_base64)
    cropped_face_np = np.frombuffer(cropped_face_buffer, np.uint8)
    cropped_face = cv2.imdecode(cropped_face_np, cv2.IMREAD_COLOR)

    # Face recognition using face_recognition library
    unknown_encoding = face_recognition.face_encodings(cropped_face)
    if not unknown_encoding:
        return "Unknown Person"

    # Compare with known face encodings
    for name, encodings in face_encodings.items():
        for encoding in encodings:
            matches = face_recognition.compare_faces([encoding], unknown_encoding[0])
            if any(matches):
                return name

    return "Unknown Person"


def get_database_names():
    return [folder for folder in os.listdir(DATABASE_PATH) if os.path.isdir(os.path.join(DATABASE_PATH, folder))]


@app.before_request
def before_request():
    app.jinja_env.globals.update(get_database_names=get_database_names)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return render_template('index.html', error='Invaid file type extension not found')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if not allowed_file(file.filename):
         return render_template('index.html', error='Invalid image type. Allowed formats are: .png, .jpg, .jpeg' )

    try:
        # Read the uploaded image
        image_stream = file.read()
        nparr = np.frombuffer(image_stream, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Use DeepFace to analyze the image
        #Options: 'opencv', 'retinaface','mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).
        results = DeepFace.analyze(img, detector_backend="mtcnn")
        
        if not results:
            return render_template('index.html', error='No faces detected in the uploaded image.')

        # Encode the original image as base64
        _, img_buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')

        # Create a list to store base64 encoded images of cropped faces
        cropped_faces = []

        for i, face_info in enumerate(results):
            # Extract face coordinates from the 'region' field
            x, y, w, h = face_info['region']['x'], face_info['region']['y'], face_info['region']['w'], face_info['region']['h']

            # Increase the size of the face rectangle by 20%
            margin = int(0.2 * min(w, h))
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(img.shape[1], x + w + margin)
            y_end = min(img.shape[0], y + h + margin)
            
            # Crop the face
            cropped_face = img[y_start:y_end, x_start:x_end]

            # Encode the cropped face as base64
            _, cropped_buffer = cv2.imencode('.png', cropped_face)
            cropped_base64 = base64.b64encode(cropped_buffer).decode('utf-8')

            # Store the base64 encoded image in the list
            cropped_faces.append(cropped_base64)

        # Face recognition using the database
        recognized_names = []
        face_encodings = load_face_encodings()

        for cropped_face in cropped_faces:
            recognized_name = recognize_person(cropped_face, face_encodings)
            recognized_names.append(recognized_name)

        return render_template('index.html', faces=results, img_base64=img_base64, cropped_faces=cropped_faces, recognized_names=recognized_names)

    except Exception as e:
        #return render_template('index.html', error=str(e))
        return render_template('index.html', error='Face is not detected in the image. Please select a new image for analyze.')

def detect_faces_in_image(image_path):
    image = cv2.imread(image_path)
    return DeepFace.analyze(image, detector_backend="mtcnn")

#Code for the 'recognize_person' and '/add_person' routes

# def analyze_image_and_detect_deepfake(image):
#     try:
#         # Use DeepFace to analyze the image
#         # Options: 'opencv', 'retinaface','mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).
#         results = DeepFace.analyze(image, detector_backend="mtcnn", actions=['emotion', 'gender', 'race'], enforce_detection=False)

#         # Deepfake detection
#         is_deepfake, confidence = is_deep_fake(image)

#         # Append deepfake detection results to the analysis results
#         results['is_fake'] = is_deepfake
#         results['confidence'] = confidence * 100  # Convert to percentage

#         return results

    # except Exception as e:
    #     return {"error": str(e)}
    
def count_faces(img):
    no=DeepFace.analyze(img,detector_backend="mtcnn",actions=('gender'))
    return len(no)

# Code for the 'recognize_person' and '/add_person' routes
@app.route('/add_person', methods=['POST'])
def add_person():
    if 'file' not in request.files or 'name' not in request.form:
        return render_template('index.html', error='Please provide both an image and a name for the person.')

    file = request.files['file']
    name = request.form['name']

    image_stream = file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if not allowed_file(file.filename):
         return render_template('index.html', error=f'Invalid file type, Trying to add: {file.filename.rsplit(".", 1)[1].lower()}. Please upload a valid image file. .png .jpg, .jpeg ' )

    try:
        # Ensure that only one face is in the image
        faces_count = count_faces(img)
        if faces_count != 1:
            return render_template('index.html', error=f'The uploaded image must contain exactly one face. Detected faces: {faces_count}')
        print(faces_count)

        # Ensure that only one image with one face is in the database for each person
        person_folder = os.path.join(DATABASE_PATH, name)

        # Create the person's folder if it doesn't exist
        os.makedirs(person_folder, exist_ok=True)

        # Generate a unique filename using UUID for the uploaded image
        unique_filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        image_path = os.path.join(person_folder, unique_filename)
        

        #file.save(image_path)
        cv2.imwrite(image_path, img)

        # Add this line to display a success message
        return render_template('index.html', message=f'Image {name} added to the data source successfully.')

    except Exception as e:
        #return render_template('index.html', error=str(e))
        return render_template('index.html', error="Face not detected in the Image. Please try again")


if __name__ == '__main__':
    app.run(debug=True)


