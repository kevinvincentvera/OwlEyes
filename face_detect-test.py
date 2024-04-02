import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

image_counter = 0  # Counter for the image filenames

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_img = frame[y:y+h, x:x+w]  # Crop the face
        img_name = f"face_{image_counter}.png"
        cv2.imwrite(img_name, face_img)  # Save the captured face
        print(f"{img_name} saved.")  # Feedback to know that an image has been saved
        image_counter += 1  # Increment the counter

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
