import os
import cv2
import time
import requests
import numpy as np
import threading
import pickle
from datetime import datetime, date
from dotenv import load_dotenv
from PIL import ImageFont, ImageDraw, Image
from imutils.video import VideoStream
import face_recognition

# === Load .env ===
load_dotenv()
MEMBERS_URL = os.getenv("MEMBERS_URL")
RTSP_URL = os.getenv("RTSP_URL")
CACHE_FILE = "face_cache.pkl"

# === Global variables ===
known_encodings = []
known_names = []
valid_to_dates = []
last_loaded_date = None
cache_lock = threading.Lock()

# === Load face encodings from cache or API ===
def load_members():
    global known_encodings, known_names, valid_to_dates, last_loaded_date

    today = date.today()
    if last_loaded_date == today:
        return

    with cache_lock:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
                if cache.get("date") == str(today):
                    print("✅ Dùng cache hôm nay.")
                    known_encodings = cache["encodings"]
                    known_names = cache["names"]
                    valid_to_dates = cache["dates"]
                    last_loaded_date = today
                    return

        print("🔄 Tải dữ liệu thành viên từ API...")
        try:
            response = requests.get(MEMBERS_URL)
            response.raise_for_status()
            members = response.json()
        except Exception as e:
            print(f"❌ Lỗi tải dữ liệu: {e}")
            return

        encodings, names, dates = [], [], []

        for idx, member in enumerate(members):
            name = member.get("name", "Unknown")
            subscription_end = member.get("endSubscriptionDate")
            images = member.get("images", [])

            for img_url in images:
                try:
                    img_data = requests.get(img_url).content
                    np_img = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                    if img is None:
                        continue

                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    enc = face_recognition.face_encodings(rgb_img)
                    if enc:
                        encodings.append(enc[0])
                        names.append(name)
                        dates.append(subscription_end)
                except Exception as e:
                    print(f"⚠️ Lỗi khi xử lý ảnh: {e}")

        known_encodings = encodings
        known_names = names
        valid_to_dates = dates
        last_loaded_date = today

        with open(CACHE_FILE, "wb") as f:
            pickle.dump({
                "date": str(today),
                "encodings": known_encodings,
                "names": known_names,
                "dates": valid_to_dates
            }, f)

        print(f"✅ Đã encode {len(known_encodings)} khuôn mặt.")

# === Drawing helper ===
def draw_text(frame, text, pos, color=(0, 255, 0)):
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("Roboto-Regular.ttf", 24)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return np.array(pil_img)

# === Face Recognition ===
def recognize_faces(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    results = []
    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
        if not known_encodings:
            continue

        distances = face_recognition.face_distance(known_encodings, enc)
        idx = np.argmin(distances)
        name = "Người lạ"
        color = (0, 0, 255)
        date_text = ""
        days_text = ""

        if distances[idx] < 0.5:
            name = known_names[idx]
            end_date_str = valid_to_dates[idx]
            try:
                end_date = datetime.strptime(end_date_str, "%m/%d/%Y")
                days_left = (end_date - datetime.now()).days
                date_text = end_date.strftime("%d/%m/%Y")
                if days_left >= 0:
                    days_text = f"Còn {days_left} ngày"
                    color = (0, 255, 0)
                else:
                    days_text = "Hết hạn"
            except:
                days_text = "Lỗi ngày"

        results.append({
            "top": top * 2, "right": right * 2,
            "bottom": bottom * 2, "left": left * 2,
            "name": name, "date": date_text,
            "days_text": days_text, "color": color
        })
    return results

# === Draw Boxes ===
def draw_faces(frame, faces):
    for f in faces:
        cv2.rectangle(frame, (f["left"], f["top"]), (f["right"], f["bottom"]), f["color"], 2)
        frame = draw_text(frame, f["name"], (f["left"], f["top"] - 45), f["color"])
        if f["date"]:
            frame = draw_text(frame, f["date"], (f["left"], f["top"] - 25), f["color"])
        if f["days_text"]:
            frame = draw_text(frame, f["days_text"], (f["left"], f["top"] - 5), f["color"])
    return frame

# === Main App ===
def main():
    print("📹 Đang kết nối camera...")
    load_members()
    stream = VideoStream(src=RTSP_URL).start()
    time.sleep(2)
    fps_time = time.time()
    frame_count = 0
    cached_faces = []

    while True:
        load_members()  # reload if day changed

        frame = stream.read()
        if frame is None:
            print("❌ Không đọc được khung hình.")
            break

        frame_count += 1
        if frame_count % 10 == 0:
            cached_faces = recognize_faces(frame)

        frame = draw_faces(frame, cached_faces)

        fps = 1 / (time.time() - fps_time)
        fps_time = time.time()
        frame = draw_text(frame, f"FPS: {int(fps)}", (10, 30), (255, 255, 0))

        cv2.imshow("🧠 Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
