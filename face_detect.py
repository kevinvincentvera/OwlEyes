import cv2
import os
from face_recog_test import recognize_face 

# start face detect
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# start video
# live_feed_url = "http://" 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device.")

# start super-resolution
sr = cv2.dnn_superres.DnnSuperResImpl_create()
model_path = "TF-ESPCN-master/export/ESPCN_x4.pb"
sr.readModel(model_path)
sr.setModel('espcn', 4)

# path to folder
known_images_folder = 'students'  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_img_gray = gray[y:y+h, x:x+w]  # crops image
        
        img_name = "detected_face_gray.png"
        cv2.imwrite(img_name, face_img_gray)  # saves cropped image
        
        # super-resolution to saved face
        face_img_gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        upsampled_face_img_gray = sr.upsample(face_img_gray)
        
        cv2.imwrite(img_name, upsampled_face_img_gray)  # save gray image
        print(f"{img_name} saved.")

        # launch comparison
        for known_image_filename in os.listdir(known_images_folder):
            known_image_path = os.path.join(known_images_folder, known_image_filename)
            recognize_face(known_image_path, img_name)  # launch face recog function from the .py

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

