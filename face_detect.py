import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_img_gray = gray[y:y+h, x:x+w]  # Crop the face from the grayscale image for better analysis
        img_name = "detected_face_gray.png"
        cv2.imwrite(img_name, face_img_gray)  # Save the grayscale face image
        print(f"{img_name} saved. Exiting program.")  # Notify the user and prepare to exit
        cv2.destroyAllWindows()  # Close all OpenCV windows
        cap.release()  # Release the webcam
        exit()  # Exit the program

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cv2.destroyAllWindows()
cap.release()

