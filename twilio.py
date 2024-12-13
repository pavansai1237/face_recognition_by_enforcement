import os
import cv2
import numpy as np
import tensorflow as tf
import mysql.connector
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array as keras_image
from sklearn.metrics.pairwise import cosine_similarity
from twilio.rest import Client

# Twilio configuration
TWILIO_ACCOUNT_SID = ""  # Replace with your Twilio Account SID
TWILIO_AUTH_TOKEN = ""  # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = ""  # Replace with your Twilio phone number
TARGET_PHONE_NUMBER = ""  # Replace with the recipient phone number

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",  # Replace with your database password
    "database": "face_reco"
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# Preprocessing function
preprocess_functions = {
    'vgg': vgg_preprocess
}

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

# Load pre-trained model
def load_model(model_name='vgg'):
    if model_name == 'vgg':
        base_model = VGG16(weights='imagenet', include_top=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def extract_features(image, model_name='vgg'):
    preprocess_function = preprocess_functions[model_name]
    target_size = (224, 224)

    image_resized = cv2.resize(image, target_size)
    image_array = keras_image(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_function(image_array)

    model = load_model(model_name)
    features = model.predict(image_array)
    return normalize_vector(features.flatten())

def detect_faces_haar(frame):
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def recognize_face(face_image, model_name='vgg'):
    extracted_feature = extract_features(face_image, model_name)

    # Fetch stored feature vectors from the database
    db_conn = get_db_connection()
    cursor = db_conn.cursor()
    cursor.execute("SELECT person_id, feature_vector FROM face_features")
    stored_features = cursor.fetchall()

    best_match = None
    highest_similarity = 0

    # Compare with stored feature vectors
    for person_id, feature_vector_str in stored_features:
        feature_vector = np.fromstring(feature_vector_str[1:-1], sep=',')  # Convert string back to array
        feature_vector = normalize_vector(feature_vector)  # Normalize stored feature vector
        similarity = cosine_similarity([extracted_feature], [feature_vector])[0][0]

        if similarity > highest_similarity and similarity > 0.85:  # Higher threshold
            highest_similarity = similarity
            cursor.execute("SELECT full_name FROM person_da WHERE id = %s", (person_id,))
            best_match = cursor.fetchone()[0]

    cursor.close()
    db_conn.close()

    return best_match if highest_similarity > 0.85 else "Unknown"  # Adjust threshold

def send_sms_via_twilio(message):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=TARGET_PHONE_NUMBER
    )

def open_camera_and_recognize_haar(model_name='vgg'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open camera. Please check camera permissions and availability.")

    recognized = False
    while not recognized:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        faces = detect_faces_haar(frame)

        for (x, y, w, h) in faces:
            face_image = frame[y:y + h, x:x + w]

            # Recognize face
            recognized_name = recognize_face(face_image, model_name)

            # Draw rectangle and display name
            color = (0, 255, 0) if recognized_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if recognized_name != "Unknown":
                print(f"Face recognized: {recognized_name}")
                send_sms_via_twilio(f"Face recognized: {recognized_name}")
                recognized = True
                break

        cv2.imshow("Live Camera - Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        open_camera_and_recognize_haar(model_name='vgg')
    except Exception as e:
        print(f"Error: {e}")
