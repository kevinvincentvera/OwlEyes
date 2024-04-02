import face_recognition

def recognize_face(known_image_path, unknown_image_path):
    print("Loading known image...")
    known_image = face_recognition.load_image_file(known_image_path)
    known_encodings = face_recognition.face_encodings(known_image)

    if not known_encodings:
        print("No faces were found in the known image.")
        return
    else:
        print(f"Detected {len(known_encodings)} face(s) in the camera feed.")

    known_encoding = known_encodings[0]

    print("Loading unknown image...")
    unknown_image = face_recognition.load_image_file(unknown_image_path)
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    if not unknown_encodings:
        print("No faces were found in the unknown image.")
        return
    else:
        print(f"Detected {len(unknown_encodings)} face(s) in the unknown image.")

    # compare the faces
    for unknown_encoding in unknown_encodings:
        results = face_recognition.compare_faces([known_encoding], unknown_encoding)
        if True in results:
            print("A matching face was found in the feed.                                Student Pass.")
        else:
            print("No matching faces were found in the feed. -----------------------------  THIS IS NOT A STUDENT.")

if __name__ == "__main__":
    known_image_path = 'kevin face.png'  
    unknown_image_path = 'detected_face_gray.png' 
    recognize_face(known_image_path, unknown_image_path)
