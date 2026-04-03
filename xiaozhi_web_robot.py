"""
Lisa Web Robot Companion - Phase 2 (Perfect Edition)
Multimodal Vision, Hands-Free Wake Word, Long-Term Memory, Emotion Reactive.
"""

import os, json, time, threading, base64
from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
import PIL.Image

# --- Vision Dependencies ---
try:
    import mediapipe as mp
    from ultralytics import YOLO
    import face_recognition
    HAS_VISION = True
except ImportError:
    HAS_VISION = False
    print("WARNING: Vision libraries missing. Run: pip install -r requirements_vision.txt")

# --- Generative AI (Gemini) ---
try:
    import google.generativeai as genai
    genai.configure(api_key="AIzaSyBasWweHJZdHoVC4Gsx295LwSZMYHNnOsc")
    
    generation_config = {"temperature": 0.5, "max_output_tokens": 150}
    model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
    memory_extractor = genai.GenerativeModel("gemini-1.5-flash", generation_config={"temperature": 0.1})
    HAS_GEMINI = True
except BaseException as e:
    HAS_GEMINI = False
    print(f"WARNING: Gemini setup failed: {e}")

# --- Optional MQTT ---
try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False

# ============================================================
# LONG-TERM MEMORY (Persistent Brain)
# ============================================================
MEMORY_FILE = "lisa_memory.json"
user_memory_facts = []

if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, 'r') as f:
            user_memory_facts = json.load(f)
    except: pass

def save_memory(fact):
    if fact and fact not in user_memory_facts:
        user_memory_facts.append(fact)
        with open(MEMORY_FILE, 'w') as f:
            json.dump(user_memory_facts, f)
        socketio.emit('memory_saved', {'fact': fact})

def check_for_memory_async(text):
    if not HAS_GEMINI: return
    try:
        prompt = f"Analyze this text: '{text}'. If the user is stating a permanent fact or preference about themselves (like a favorite color, their name, their job, something they own), extract it as a single concise fact sentence (e.g. 'User's favorite color is blue'). If no fact is present, output exactly the word 'NONE'."
        res = memory_extractor.generate_content(prompt)
        res_text = res.text.strip()
        if res_text and "NONE" not in res_text.upper():
            save_memory(res_text)
    except Exception as e:
        print(f"Memory extraction error: {e}")

def get_chat_session():
    if not HAS_GEMINI: return None
    sys_prompt = "You are Lisa, a friendly, concise AI robot companion. Keep responses short. "
    if user_memory_facts:
        sys_prompt += "Here are facts you know about the user: " + "; ".join(user_memory_facts) + ". "
    return model.start_chat(history=[
        {"role": "user", "parts": [sys_prompt]},
        {"role": "model", "parts": ["I understand! I'm Lisa, your friendly AI companion."]}
    ])

if HAS_GEMINI:
    chat_session = get_chat_session()
else:
    chat_session = None

# ============================================================
# VISION STATE & CONFIG
# ============================================================
class RobotState:
    pan_angle = 90
    tilt_angle = 90
    latest_frame = None  # RGB Frame for Gemini Multimodal
    last_emotion = "neutral"
    last_concern_time = 0

class Config:
    frame_center_x = 320
    frame_center_y = 240
    servo_gain = 0.05
    gesture_cooldown = 5.0
    face_match_tolerance = 0.6
    mqtt_topic = "home/light/set"

state = RobotState()
cfg = Config()

# Vision Models
mp_hands = None
hands_detector = None
mp_face_mesh = None
face_mesh_detector = None
mp_draw = None
yolo_model_net = None
known_encoding = None

if HAS_VISION:
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_detector = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    
    mp_draw = mp.solutions.drawing_utils
    yolo_model_net = YOLO('yolov8n.pt')
    
    owner_path = "owner.jpg"
    if os.path.exists(owner_path):
        try:
            img = face_recognition.load_image_file(owner_path)
            encodings = face_recognition.face_encodings(img)
            if encodings: known_encoding = encodings[0]
        except: pass

def _analyze_emotion_facemesh(face_landmarks):
    """
    Very crude heuristic to determine smile vs frown using FaceMesh landmarks.
    Mouth left corner: 61, Mouth right corner: 291
    Bottom lip center: 14, Top lip center: 13
    """
    try:
        left = face_landmarks.landmark[61].y
        right = face_landmarks.landmark[291].y
        center = face_landmarks.landmark[14].y
        
        corner_avg = (left + right) / 2.0
        diff = center - corner_avg 
        # If mouth corners are significantly higher (lower Y value) than bottom lip -> Smile
        if diff > 0.02: return "happy"
        # If mouth corners are significantly lower than bottom lip -> Frown/Sad
        elif diff < -0.01: return "sad"
    except: pass
    return "neutral"

# ============================================================
# MQTT SMART HOME CONTROLLER
# ============================================================
class SmartHomeController:
    def __init__(self):
        self.client = None
        self.connected = False
    def connect(self, broker, port=1883, username="", password=""):
        if not HAS_MQTT: return False, "paho-mqtt not installed"
        try:
            if self.client: self.disconnect()
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            if username: self.client.username_pw_set(username, password)
            self.client.on_connect = lambda c,u,f,r,p=None: setattr(self, 'connected', r==0)
            self.client.on_disconnect = lambda c,u,f,r,p=None: setattr(self, 'connected', False)
            self.client.connect(broker, int(port), 60)
            self.client.loop_start()
            time.sleep(1)
            return self.connected, "Connected" if self.connected else "Connection failed"
        except Exception as e:
            return False, str(e)
    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.client, self.connected = None, False
    def publish(self, topic, payload):
        if not self.connected or not self.client: return False
        try:
            self.client.publish(topic, payload)
            return True
        except: return False

# ============================================================
# FLASK APP & WEBSOCKETS
# ============================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'lisa-robot-2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', max_http_buffer_size=5000000)
smart_home = SmartHomeController()

def trigger_speech(text):
    socketio.emit('speak_text', {'text': text})

@app.route('/')
def index():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_ui.html")
    with open(path, 'r', encoding='utf-8') as f:
        return render_template_string(f.read())

@socketio.on('voice_command')
def handle_voice_command(data):
    global chat_session
    text = data.get('text', '').strip()
    if not text: return
    print(f"\nUser: {text}")
    
    # 1. Start Memory extraction in background
    threading.Thread(target=check_for_memory_async, args=(text,)).start()
    
    # 2. Check for multimodal vision cue ("see", "look", "what is")
    lower = text.lower()
    multimodal_parts = [text]
    
    if any(w in lower for w in ["see", "look", "what", "holding", "camera"]):
        if state.latest_frame is not None:
            pil_img = PIL.Image.fromarray(state.latest_frame)
            multimodal_parts = [pil_img, text]
            print("[Sent image to Gemini]")
    
    # 3. Check for specific hardcoded commands first (MQTT / Spotify etc)
    mqtt_resp = None
    if smart_home.connected:
        if "turn on" in lower:
            ok = smart_home.publish(cfg.mqtt_topic, "ON")
            mqtt_resp = "I have turned the light on for you." if ok else "MQTT command failed."
        elif "turn off" in lower:
            ok = smart_home.publish(cfg.mqtt_topic, "OFF")
            mqtt_resp = "I have turned the light off." if ok else "MQTT command failed."

    # 4. Ask Gemini
    if mqtt_resp:
        response = mqtt_resp
    elif HAS_GEMINI:
        try:
            res = chat_session.send_message(multimodal_parts)
            response = res.text
        except Exception as e:
            response = "I had a cognitive glitch while processing that."
            print(f"Gemini error: {e}")
            # If token limit hits or generic error, reset chat session
            chat_session = get_chat_session()
    else:
        response = "I heard you, but my AI cloud modules are offline."

    print(f"Lisa: {response}")
    trigger_speech(response)

@socketio.on('video_frame')
def handle_video_frame(data):
    if not HAS_VISION: return
    encoded_data = data.split(',')[1] if ',' in data else data
    try:
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return
    except: return

    global known_encoding
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    state.latest_frame = rgb # Cache for multimodal Gemini
    now = time.time()
    
    # 1. Emotion Detection via FaceMesh
    face_results = face_mesh_detector.process(rgb)
    if face_results.multi_face_landmarks:
        geomotion = _analyze_emotion_facemesh(face_results.multi_face_landmarks[0])
        if geomotion != state.last_emotion:
            state.last_emotion = geomotion
            socketio.emit('emotion_update', {'emotion': geomotion})
            
            # Proactive intelligence: If sad for a while, trigger concern
            if geomotion == "sad" and (now - state.last_concern_time > 60):
                state.last_concern_time = now
                threading.Timer(2.0, lambda: trigger_speech("Hey, you look a bit stressed. Is everything okay? I'm here if you need to talk.")).start()

    # 2. Hands Gesture (Peace sign)
    hand_results = hands_detector.process(rgb)
    if hand_results.multi_hand_landmarks:
        for hand_lm in hand_results.multi_hand_landmarks:
            # We don't draw dots to keep HUD clean in Phase 2
            if _is_peace_sign(hand_lm) and (now - getattr(state, 'last_gesture', 0) > 5.0):
                print("✌️ Peace sign detected!")
                state.last_gesture = now
                trigger_speech("Peace sign recognized!")

    # 3. Face Registration & Intruder YOLO
    if known_encoding is None:
        face_locs = face_recognition.face_locations(rgb)
        if face_locs:
            face_encs = face_recognition.face_encodings(rgb, face_locs)
            if face_encs:
                known_encoding = face_encs[0]
                try: cv2.imwrite("owner.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                except: pass
                trigger_speech("Identity confirmed. I will recognize you from now on.")

    # Frame processing loop (Tactical HUD)
    results = yolo_model_net(frame, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            new_pan  = state.pan_angle  - int((cx - cfg.frame_center_x) * cfg.servo_gain)
            new_tilt = state.tilt_angle + int((cy - cfg.frame_center_y) * cfg.servo_gain)
            state.pan_angle  = max(20, min(160, new_pan))
            state.tilt_angle = max(20, min(160, new_tilt))

            name, color = "UNKNOWN", (0, 60, 255)
            if known_encoding is not None:
                floc = [(max(0,y1), min(frame.shape[1],x2), min(frame.shape[0],y2), max(0,x1))]
                fencs = face_recognition.face_encodings(rgb, floc)
                if fencs:
                    match = face_recognition.compare_faces([known_encoding], fencs[0], cfg.face_match_tolerance)
                    if match[0]: name, color = "OWNER", (0, 210, 60)
                    else: name = "INTRUDER"
            else:
                name, color = "SCANNING", (0, 165, 255)

            # Draw HUD
            blen = 15
            for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame, (px,py), (px+dx*blen, py), color, 2)
                cv2.line(frame, (px,py), (px, py+dy*blen), color, 2)
            cv2.putText(frame, name, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"P:{state.pan_angle} T:{state.tilt_angle}", (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Return frame to client
    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ret:
        b64_str = base64.b64encode(buf).decode('utf-8')
        emit('video_response', {'image': 'data:image/jpeg;base64,' + b64_str})

@socketio.on('mqtt_connect')
def mqtt_connect(data):
    ok, msg = smart_home.connect(data.get('broker'), data.get('port',1883), data.get('username'), data.get('password'))
    emit('mqtt_status', {'connected': ok, 'message': msg})

if __name__ == '__main__':
    port = 5000
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token("3BrHBtNHD8JXtrdlouC9zckYhBF_vbxibADFwULGWPxnqvGm")
        public_url = ngrok.connect(port).public_url
        print(f"\n========================================================")
        print(f"🌟 NGROK TUNNEL ACTIVE!")
        print(f"🌍 Access Lisa from anywhere at: {public_url}")
        print(f"========================================================\n")
    except ImportError:
        print("\n(Optional) Install 'pyngrok' (pip install pyngrok) to expose the dashboard to the public internet.\n")
        
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
