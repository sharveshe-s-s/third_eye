import os
import time
import threading
import base64
import cv2
import numpy as np
import pyaudio
import struct
import gc
import pvporcupine
from openai import OpenAI

# --- 1. USER CONFIGURATION ---
OPENAI_KEY = "enter your open ai api key"
PICOVOICE_KEY = "enter your picovoice key"
KEYWORD_FILE_PATH = "add your key word path"
MODEL_PATH = "add your model path here"
LABEL_PATH = "add your label path here"
IMAGE_PATH = "view.jpg"

# AUDIO HARDWARE SETTINGS (CRITICAL FOR PI ZERO)
MIC_RATE = 48000            # Your hardware is locked here
PORCUPINE_RATE = 16000      # AI needs this
RESAMPLE_RATIO = 3          # 48000 / 16000 = 3
MIC_DEVICE_KEYWORD = "AB13X"

# UPDATED HAZARD LIST (INDOOR + OUTDOOR)
CONFIDENCE_THRESHOLD = 0.45 # Lowered slightly for indoor lighting
HAZARD_OBJECTS = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
    'stop sign', 'stairs', 'hole', 'fire hydrant',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- 2. GLOBAL VARIABLES ---
client = OpenAI(api_key=OPENAI_KEY)
camera_lock = threading.Lock()
speaking_lock = threading.Lock()
latest_frame = None
stop_threads = False
is_describing = False

# --- 3. UTILITIES ---
def speak(text, vital=False):
    """
    Uses OpenAI TTS (High Quality). Falls back to espeak if offline.
    """
    if not text: return

    # Don't overlap vital warnings (like 'Stop!')
    if vital and speaking_lock.locked():
        return

    with speaking_lock:
        print(f"🗣️ Jarvis: {text}")
        try:
            # 1. Generate MP3 from OpenAI
            response = client.audio.speech.create(
                model="tts-1",
                voice="onyx", # Options: alloy, echo, fable, onyx, nova, shimmer
                input=text
            )

            # 2. Save to temp file
            speech_file = "speech.mp3"
            response.stream_to_file(speech_file)

            # 3. Play using hardware acceleration (mpg123)
            # -a plughw:0,0 directs it to your USB headset
            os.system(f"mpg123 -a plughw:0,0 -q {speech_file}")

        except Exception as e:
            print(f"⚠️ TTS Error (Using Backup): {e}")
            # Fallback to Robot Voice if internet fails
            cmd = f'espeak -v en-us+m3 -p 40 -s 150 "{text}" --stdout | aplay -D plughw:0,0 2>/dev/null'
            os.system(cmd)

def find_mic_index(pa):
    """Finds the AB13X mic index automatically"""
    count = pa.get_device_count()
    for i in range(count):
        try:
            info = pa.get_device_info_by_index(i)
            if MIC_DEVICE_KEYWORD in info.get('name', ''):
                print(f"✅ Found Mic '{info['name']}' at Index {i}")
                return i
        except:
            continue
    print("⚠️ Mic not found by name. Defaulting to Index 0")
    return 0

def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

# --- 4. HAZARD DETECTION LOOP (THREAD 1) ---
def hazard_detection_loop():
    global latest_frame, stop_threads, is_describing

    # Import TFLite inside thread to save main-thread memory
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except:
            import tensorflow.lite as Interpreter

    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        return

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]

    labels = load_labels(LABEL_PATH)

    # Camera Setup (Low Res for Speed)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"👁️ VISION THREAD ONLINE")
    last_warning_time = {}
    frame_counter = 0

    while not stop_threads:
        # Pause vision if we are Describing (Save bandwidth/CPU)
        if is_describing:
            time.sleep(0.5)
            continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        with camera_lock:
            latest_frame = frame.copy()

        # THROTTLING: Run TFLite every 3rd frame (Prevents Audio Stutter)
        frame_counter += 1
        if frame_counter % 3 != 0:
            continue

        try:
            image_resized = cv2.resize(frame, (w, h))
            input_data = np.expand_dims(image_resized, axis=0)

            if input_details[0]['dtype'] == np.float32:
                input_data = (np.float32(input_data) - 127.5) / 127.5

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            scores = interpreter.get_tensor(output_details[2]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0]
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]

            current_time = time.time()

            for i in range(len(scores)):
                if scores[i] > CONFIDENCE_THRESHOLD:
                    class_id = int(classes[i])
                    obj_name = labels.get(class_id, "Unknown")

                    if obj_name in HAZARD_OBJECTS:
                        # Estimate distance by box height
                        ymin, xmin, ymax, xmax = boxes[i]
                        obj_height = ymax - ymin
                        is_close = obj_height > 0.50 # 50% of screen height

                        # Debounce (Don't spam)
                        cooldown = 4.0 if is_close else 8.0

                        if current_time - last_warning_time.get(obj_name, 0) > cooldown:
                            prefix = "Stop. " if is_close else ""
                            # Only speak hazards if OpenAI isn't talking
                            if not speaking_lock.locked():
                                speak(f"{prefix}{obj_name}", vital=True)
                                last_warning_time[obj_name] = current_time
                                break # Only speak one hazard at a time

        except Exception as e:
            print(f"Vision Error: {e}")

    cap.release()

# --- 5. SCENE DESCRIPTION (OPENAI) ---
def describe_scene():
    global is_describing
    is_describing = True

    # Audio Feedback
    # We use os.system directly for the 'beep' to avoid network lag
    os.system("espeak 'Analyzing' -s 200 >/dev/null 2>&1")

    target_frame = None
    with camera_lock:
        if latest_frame is not None:
            target_frame = latest_frame.copy()

    if target_frame is None:
        speak("Camera not ready.")
        is_describing = False
        return

    try:
        _, buffer = cv2.imencode('.jpg', target_frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "I am blind. Describe the scene in one natural sentence. Mention navigation hazards or money value if present."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]}
            ],
            max_tokens=80
        )

        answer = response.choices[0].message.content
        speak(answer)

    except Exception as e:
        print(f"OpenAI Error: {e}")
        speak("Connection failed.")

    is_describing = False
    gc.collect()
    print("👁️ VISION RESUMED")

# --- 6. AUDIO LOOP (MAIN THREAD) ---
def audio_loop():
    global stop_threads

    porcupine = pvporcupine.create(
        access_key=PICOVOICE_KEY,
        keyword_paths=[KEYWORD_FILE_PATH],
        sensitivities=[1.0])

    pa = pyaudio.PyAudio()
    mic_index = 0

    # RESAMPLING SETUP
    read_chunk = porcupine.frame_length * RESAMPLE_RATIO

    try:
        stream = pa.open(
            rate=MIC_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=read_chunk,
            input_device_index=mic_index
        )

        print(f"👂 LISTENING: 48kHz -> Resampled to 16kHz")

        while not stop_threads:
            if is_describing:
                time.sleep(0.1)
                continue

            try:
                # 1. READ
                pcm = stream.read(read_chunk, exception_on_overflow=False)

                # 2. UNPACK & RESAMPLE
                pcm_unpacked = struct.unpack_from("h" * read_chunk, pcm)
                pcm_16k = pcm_unpacked[::RESAMPLE_RATIO]

                # 3. DETECT
                result = porcupine.process(pcm_16k)

                if result >= 0:
                    print("✨ EDITH DETECTED ✨")
                    describe_scene()

            except OSError:
                pass

    except Exception as e:
        print(f"CRITICAL AUDIO ERROR: {e}")
    finally:
        if 'stream' in locals(): stream.close()
        pa.terminate()
        porcupine.delete()

# --- 7. MAIN ENTRY ---
if __name__ == "__main__":
    t_vision = threading.Thread(target=hazard_detection_loop)
    t_vision.daemon = True
    t_vision.start()

    try:
        audio_loop()
    except KeyboardInterrupt:
        print("\nStopping Third Eye...")
        stop_threads = True
        time.sleep(1)
