import cv2
import face_recognition

# Load the reference image
reference_image = face_recognition.load_image_file("reference.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]  # Extract facial features

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = frame[:, :, ::-1]

    # Detect faces and extract encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the captured face with the reference image
        match = face_recognition.compare_faces([reference_encoding], face_encoding, tolerance=0.5)
        
        # Get coordinates of detected face
        top, right, bottom, left = face_location
        color = (0, 255, 0) if match[0] else (0, 0, 255)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Display result
        label = "Match" if match[0] else "No Match"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the video frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
video_capture.release()
cv2.destroyAllWindows()
