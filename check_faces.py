import cv2
import face_recognition

# === Replace this with your RTSP stream ===
rtsp_url = "rtsp://admin:An29122001@192.168.1.15:554/cam/realmonitor?channel=1&subtype=1"

# Open RTSP stream
video_capture = cv2.VideoCapture(rtsp_url)

# Load a known face (e.g., yourself)
known_image = face_recognition.load_image_file("an.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]
known_names = ["An"]
known_encodings = [known_encoding]

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Không lấy được frame từ camera.")
        break

    # Resize frame for speed (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations & encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Scale back up face locations (since we resized earlier)
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw rectangle + name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('RTSP Face Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
