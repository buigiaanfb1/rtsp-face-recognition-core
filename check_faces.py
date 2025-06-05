import os
from dotenv import load_dotenv
import cv2
import face_recognition
import requests
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime

# Load environment variables
load_dotenv()
MEMBERS_URL = os.getenv("MEMBERS_URL")
RTSP_URL = os.getenv("RTSP_URL")

# === RTSP stream ===
video_capture = cv2.VideoCapture(RTSP_URL)

# === Load member data ===
response = requests.get(MEMBERS_URL)
members = response.json()

known_encodings = []
known_names = []
valid_to_dates = []

for member in members:
    for image_url in member["images"]:
        image_data = requests.get(image_url).content
        np_img = np.asarray(bytearray(image_data), dtype=np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(rgb_img)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(member["name"])
            valid_to_dates.append(member.get("endSubscriptionDate"))  # could be None

# === Helper to draw text in Vietnamese ===
def draw_text_with_pil(frame, text, position, color=(0, 255, 0)):
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("Roboto-Regular.ttf", 24)  # Make sure it's in the same folder
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return np.array(pil_img)

# === Main loop ===
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Không lấy được frame từ camera.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Người lạ"
        color = (0, 0, 255)
        end_date_str = ""
        days_left_text = ""

        if True in matches:
            index = matches.index(True)
            name = known_names[index]
            end_date_str = valid_to_dates[index]
            color = (0, 255, 0)

            # Check expiration
            if end_date_str:
                try:
                    end_date = datetime.strptime(end_date_str, "%m/%d/%Y")
                    formatted_end_date = end_date.strftime("%d/%m/%Y")
                    now = datetime.now()
                    days_left = (end_date - now).days
                    days_left_text = f"Còn {days_left} ngày" if days_left >= 0 else "Hết hạn"
                    if days_left < 0:
                        color = (0, 0, 255)
                except Exception as e:
                    print("Date parse error:", e)

        # Scale back up
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        frame = draw_text_with_pil(frame, f"{name}", (left, top - 40), color)
        if end_date_str:
            frame = draw_text_with_pil(frame, f"{end_date_str}", (left, top - 20), color)
        if days_left_text:
            frame = draw_text_with_pil(frame, days_left_text, (left, top), color)

    cv2.imshow("RTSP Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
