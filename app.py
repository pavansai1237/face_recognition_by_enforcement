import os
import cv2
import numpy as np
import random
import tensorflow as tf
import mysql.connector
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array as keras_image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "pavan",  # Replace with your database password
    "database": "face_reco"
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# Preprocessing function mapper
preprocess_functions = {
    'vgg': vgg_preprocess,
    'resnet': resnet_preprocess,
    'densenet': densenet_preprocess
}

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

# Create a directory if it doesn't exist
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# Convert image to binary format
def image_to_blob(image):
    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes()

# Load pre-trained models
def load_model(model_name='vgg'):
    if model_name == 'vgg':
        base_model = VGG16(weights='imagenet', include_top=False)
    elif model_name == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False)
    elif model_name == 'densenet':
        base_model = DenseNet121(weights='imagenet', include_top=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Function to augment images
def augment_image(image):
    augmented_images = []

    # Change Hair Color
    img_color_change = image.copy()
    img_color_change = cv2.cvtColor(img_color_change, cv2.COLOR_BGR2HSV)
    img_color_change[..., 0] = (img_color_change[..., 0] + random.randint(10, 40)) % 180
    img_color_change = cv2.cvtColor(img_color_change, cv2.COLOR_HSV2BGR)
    augmented_images.append(img_color_change)

    # Change Brightness
    alpha = random.uniform(0.5, 1.5)
    beta = random.randint(-50, 50)
    img_brightness = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    augmented_images.append(img_brightness)

    # Add Noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    img_noise = cv2.add(image, noise)
    img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)
    augmented_images.append(img_noise)

    # Rotate Image
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.randint(-10, 10), 1)
    img_rotated = cv2.warpAffine(image, M, (cols, rows))
    augmented_images.append(img_rotated)

    return augmented_images

# Extract features using the model's built-in methods
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

# Enroll a face and store augmented images in the database and a folder
def enroll_face(image, person_name, model_name='vgg'):
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")

    (x, y, w, h) = faces[0]
    face_image = image[y:y + h, x:x + w]

    # Augment the image
    augmented_images = augment_image(face_image)

    # Save augmented images to a folder
    output_dir = f"augmented_images/{person_name}"
    create_directory(output_dir)

    for i, aug_img in enumerate(augmented_images):
        file_path = os.path.join(output_dir, f"{person_name}_aug_{i + 1}.jpg")
        cv2.imwrite(file_path, aug_img)

    # Save person details and augmented images to the database
    db_conn = get_db_connection()
    cursor = db_conn.cursor()

    # Insert person into `person_da`
    cursor.execute("INSERT INTO person_da (full_name) VALUES (%s)", (person_name,))
    person_id = cursor.lastrowid

    # Insert images into `face_pho`
    for aug_img in augmented_images:
        img_blob = image_to_blob(aug_img)
        cursor.execute("INSERT INTO face_pho (person_id, image) VALUES (%s, %s)", (person_id, img_blob))

    db_conn.commit()

    # Extract features for each augmented image and insert them into the `face_features` table
    features_list = []
    for aug_img in augmented_images:
        features = extract_features(aug_img, model_name)
        features_list.append(features)

        # Insert features into the `face_features` table
        cursor.execute("INSERT INTO face_features (person_id, feature_vector) VALUES (%s, %s)", 
                       (person_id, str(features.tolist())))

    db_conn.commit()

    cursor.close()
    db_conn.close()

    return person_name, features_list

# Detect faces using Haar cascades
def detect_faces_haar(frame):
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# Function to recognize faces based on feature comparison
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

# Open camera and recognize faces in real-time
def open_camera_and_recognize_haar(model_name='vgg'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open camera. Please check camera permissions and availability.")

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        faces = detect_faces_haar(frame)

        for (x, y, w, h) in faces:
            face_image = frame[y:y + h, x:x + w]

            # Recognize face
            recognized_name = recognize_face(face_image, model_name)
            label = recognized_name if recognized_name else "Unknown"

            # Draw rectangle around detected face and display the name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Live Camera - Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Simulate an uploaded image (replace with actual file upload logic)
    uploaded_image = cv2.imread("E:\\pranaysir1.jpg")  # Replace with input image
    person_name = "pranaysir"

    try:
        # Enroll a face
        person_name, features = enroll_face(uploaded_image, person_name)
        print(f"Enrolled {person_name} successfully with {len(features)} feature sets.")

        # Open live camera for recognition
        open_camera_and_recognize_haar(model_name='vgg')
    except Exception as e:
        print(f"Error: {e}")