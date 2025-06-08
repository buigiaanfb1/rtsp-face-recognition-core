import os
import cv2
import time
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
from imutils.video import VideoStream
import face_recognition

# === Load .env ===
load_dotenv()
MEMBERS_URL = os.getenv("MEMBERS_URL")
RTSP_URL = os.getenv("RTSP_URL")

# === Load known encodings ===
known_encodings = []
known_names = []
valid_to_dates = []

def load_members():
    print("🔄 Đang lấy thông tin thành viên...")

    try:
        response = requests.get(MEMBERS_URL)
        response.raise_for_status()
        members = response.json()
        print(f"✅ Đã lấy được thông tin của {len(members)} thành viên từ API.")
    except Exception as e:
        print(f"❌ Lấy thông tin thành viên thất bại: {e}")
        return

    total_images = 0
    encoded_count = 0

    for idx, member in enumerate(members):
        name = member.get("name", "Unknown")
        subscription_end = member.get("endSubscriptionDate")
        images = member.get("images", [])

        print(f"📦 [{idx+1}/{len(members)}] Đang xử lí thành viên: {name} ({len(images)} ảnh)")

        for img_idx, img_url in enumerate(images):
            total_images += 1
            try:
                img_data = requests.get(img_url).content
                np_img = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"  ⚠️ Decode ảnh thất bại: {img_url}")
                    continue

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_img)

                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(name)
                    valid_to_dates.append(subscription_end)
                    encoded_count += 1
                    print(f"  ✅ Ảnh {img_idx+1} encoding thành công.")
                else:
                    print(f"  ⚠️  Ảnh {img_idx+1} không tìm thấy mặt.")

            except Exception as e:
                print(f"  ❌ Có lỗi trong quá trình xử lí ảnh {img_url}: {e}")


load_members()

# === Drawing helper ===
def draw_text_with_pil(frame, text, position, color=(0, 255, 0)):
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("Roboto-Regular.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return np.array(pil_img)

# === Constants ===
THRESHOLD = 0.5
FRAME_SKIP = 10
frame_count = 0
cached_results = []

# === Face recognition logic ===
def recognize_faces(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    results = []
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, encoding)
        if len(distances) == 0:
            name = "Người lạ"
            color = (0, 0, 255)
            end_date_str = ""
            days_left_text = ""
        else:
            best_index = np.argmin(distances)
            best_distance = distances[best_index]
            if best_distance < THRESHOLD:
                name = known_names[best_index]
                end_date_str = valid_to_dates[best_index]
                color = (0, 255, 0)
                days_left_text = ""
                if end_date_str:
                    try:
                        end_date = datetime.strptime(end_date_str, "%m/%d/%Y")
                        now = datetime.now()
                        days_left = (end_date - now).days
                        days_left_text = f"Còn {days_left} ngày" if days_left >= 0 else "Hết hạn"
                        if days_left < 0:
                            color = (0, 0, 255)
                    except Exception as e:
                        print("Date parse error:", e)
            else:
                name = "Người lạ"
                color = (0, 0, 255)
                end_date_str = ""
                days_left_text = ""

        results.append({
            "top": top * 2,
            "right": right * 2,
            "bottom": bottom * 2,
            "left": left * 2,
            "name": name,
            "end_date": end_date_str,
            "days_left_text": days_left_text,
            "color": color
        })
    return results

# === Drawing results ===
def draw_faces(frame, faces):
    for face in faces:
        top, right, bottom, left = face["top"], face["right"], face["bottom"], face["left"]
        name, end_date, days_left_text = face["name"], face["end_date"], face["days_left_text"]
        color = face["color"]

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        frame = draw_text_with_pil(frame, name, (left, top - 40), color)
        if end_date:
            frame = draw_text_with_pil(frame, end_date, (left, top - 20), color)
        if days_left_text:
            frame = draw_text_with_pil(frame, days_left_text, (left, top), color)
    return frame

# === Main loop ===
def main():
    global frame_count, cached_results

    video_capture = VideoStream(src=RTSP_URL).start()
    time.sleep(2.0)  # Warm-up stream
    prev_time = time.time()

    while True:
        frame = video_capture.read()
        if frame is None:
            print("Can't read frame from camera")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP == 0:
            cached_results = recognize_faces(frame)

        frame = draw_faces(frame, cached_results)

        # === Optional: Show FPS ===
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        frame = draw_text_with_pil(frame, f"FPS: {int(fps)}", (10, 30), (255, 255, 0))

        cv2.imshow("Gym Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video_capture.stop()

if __name__ == "__main__":
    main()
