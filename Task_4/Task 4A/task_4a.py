import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from scipy.ndimage import median_filter

combat = "combat"
rehab = "human_aid_rehabilitation"
military_vehicles = "military_vehicles"
fire = "fire"
destroyed_building = "destroyed_buildings"

identified_labels = {}  
i = 0
check = 0

def detect_aruco_markers(image):

    # Initialize the ArUco dictionary and parameters
    dictionary = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    if image is None:
        print("Error: Could not read the image. Please check the image path.")
    
    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
    
    # Check if any markers were found
    if ids is not None:
        # Loop through the detected markers and find IDs 5, 7, 4, and 6
        marker_5 = None
        marker_7 = None
        marker_4 = None
        marker_6 = None

        for i, marker_id in enumerate(ids):
            if marker_id == 5:
                marker_5 = corners[i][0]
            elif marker_id == 7:
                marker_7 = corners[i][0]
            elif marker_id == 4:
                marker_4 = corners[i][0]
            elif marker_id == 6:
                marker_6 = corners[i][0]

        # Check if all markers were found
        if marker_5 is not None and marker_7 is not None and marker_4 is not None and marker_6 is not None:
            # Get the coordinates of the marker corners
            coord_5 = marker_5[0]  # Bottom right corner of marker 5
            coord_7 = marker_7[3]  # Top right corner of marker 7
            coord_4 = marker_4[1]  # Bottom left corner of marker 4
            coord_6 = marker_6[2]  # Top left corner of marker 6

            # Define the width of the output square
            output_size = 700  # Adjust as needed

            # Calculate the transformation matrix to perform perspective transform
            src = np.array([coord_5, coord_7, coord_4, coord_6], dtype=np.float32)
            dst = np.array([[output_size - 1, output_size - 1], [output_size - 1, 0], [0, output_size - 1], [0, 0]], dtype=np.float32)

            transform_matrix = cv2.getPerspectiveTransform(src, dst)

            # Perform perspective transform to crop and reshape
            transformed_image = cv2.warpPerspective(image, transform_matrix, (output_size, output_size))
            flipped_image = cv2.flip(transformed_image, 0)
            mirrored_image = cv2.flip(flipped_image, 1)


            # Display the transformed image

            return mirrored_image
        
    return None

def classify_event(image):

    model_path = 'C:/Users/Rudraksh/OneDrive/Desktop/Task_2C/my_model_d.h5'
    model = load_model(model_path)

    # noise_factor = 5
    # noisy_image = image + np.random.normal(0, noise_factor, image.shape)

    # # Clip the pixel values to be in the valid range [0, 255]
    # noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    # alpha = 1.3
    # beta = 1.1
    # adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # blurred = cv2.GaussianBlur(adjusted, (0, 0), 2)
    # sharpened = cv2.addWeighted(adjusted, 1.5, blurred, -0.5, 0)

    img = cv2.resize(image, (64, 64))

    # cv2.imshow('filter', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype='float32') / 255.0
    img = np.expand_dims(img, axis=0)

    with tf.device('/CPU:0'):  # Use CPU to avoid potential GPU memory issues
        y_pred = model.predict(img)

    y_pred = np.argmax(y_pred, axis=1)
    class_names = ["combat", "destroyed_buildings", 'fire', "human_aid_rehabilitation", "military_vehicles"]
    class_names_label = {i: class_name for i, class_name in enumerate(class_names)}

    event = class_names_label[y_pred[0]]

    return event

def task_4a_return(image):

    w = 56
    h = 56

    rectangles = [
            (144, 606),  # Area 1
            (464, 470),  # Area 2
            (470, 331),  # Area 3
            (135, 330),  # Area 4
            (148, 104)   # Area 5
        ]

    event_list = []

    for i, (x, y) in enumerate(rectangles):
        if y + h <= image.shape[0] and x + w <= image.shape[1]:
            cropped = image[y:y + h, x:x + w]
            event_list.append(cropped)

    for img_index in range(0,5):
        img = event_list[img_index]

        #cv2.imshow("img", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        global identified_labels

        detected_event = classify_event(img)
        print((img_index + 1), detected_event)
        if detected_event == combat:
            identified_labels.update({chr(65+img_index): 'combat'})
        if detected_event == rehab:
           identified_labels.update({chr(65+img_index): 'human_aid_rehabilitation'})
        if detected_event == military_vehicles:
            identified_labels.update({chr(65+img_index): 'military_vehicles'})
        if detected_event == fire:
            identified_labels.update({chr(65+img_index): 'fire'})
        if detected_event == destroyed_building:
            identified_labels.update({chr(65+img_index): 'destroyed_buildings'})
    
    global check
    check = 100
    time.sleep(5)
    check = 1000
    return identified_labels

if __name__ == "__main__":
    desired_camera_index = 1  # Replace this with the correct index for your USB camera
    cap = cv2.VideoCapture(desired_camera_index, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 1) #Set brightness to 1

    while True:
        ret, frame = cap.read()  # Read a frame from the camera

        if not ret:
            print("Failed to capture frame")

        mirrored_image = detect_aruco_markers(frame)
        thread = threading.Thread(target=task_4a_return, args=(mirrored_image,))

        if check > 250:
            break

        if check > 150 and check < 250:
            color = (0, 255, 0)
            w = 56
            h = 56
            rectangles = [
                (144, 606),  # Area 1
                (463, 470),  # Area 2
                (468, 332),  # Area 3
                (134, 330),  # Area 4
                (148, 104)   # Area 5
            ]
            for (x, y), (label, text) in zip(rectangles, identified_labels.items()):
                cv2.rectangle(mirrored_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(mirrored_image, text, (x - 25, y - 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        if mirrored_image is not None:
            cv2.imshow('Arena Feed', mirrored_image)
            if i < 1:
                thread.start()
                i = 3

        if i > 2 and i < 4:
            if check > 50 and check < 150:
                # print("HELLO, WORLD")
                print("identified_labels = ",identified_labels)
                check = 200
                i = 7

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
            break

    cap.release()
    cv2.destroyAllWindows()