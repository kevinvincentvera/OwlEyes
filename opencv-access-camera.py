import cv2


rtsp_url = 'rtsp://username:password@camera_ip_address'


# Capture video from the first camera device
cap = cv2.VideoCapture(rtsp_url)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Display the resulting frame
    cv2.imshow('Security Camera Feed', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

