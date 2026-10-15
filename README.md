# Third Eye: AI-Powered Smart Glasses for the Visually Impaired

![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%20Zero%202%20W-c51a4a)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20development-yellow)

## Project Overview
Third Eye is a low-cost, hands-free assistive wearable for the visually impaired, built on the Raspberry Pi Zero 2 W (512MB RAM). It uses a **Hybrid Edge-Cloud Architecture** to balance physical safety with deep cognitive understanding:

* **Offline Edge Safety (<50ms target):** A local TensorFlow Lite model continuously scans for immediate hazards and alerts the user without needing internet.
* **Cloud Intelligence (~1.2s perceived latency):** GPT-4o Vision and the Whisper API handle complex tasks such as verbatim document reading, currency identification, and scene description.
* **Edge-Bypass Router (<50ms):** Routine queries (time, battery level, system status, navigation toggles) are answered locally with no cloud API call.
* **Background Audio Engine:** Streams YouTube audio asynchronously using `yt-dlp` and `mpv` without interrupting hazard detection.

Third Eye **complements traditional mobility aids** (such as the white cane or a guide dog) by providing active overhead hazard alerts and cognitive assistance. It is not a replacement for them.

> **Safety notice:** Third Eye is an assistive prototype, not a certified medical or safety device. Cloud AI output (including prescription labels, currency, and scene descriptions) can be wrong. Never rely on it alone for medication, financial, or navigation-critical decisions.

---

## Key Features

### 1. Low-Latency Local Safety (<50ms)
* **Technology:** TensorFlow Lite (quantized SSD MobileNet V2)
* **Function:** Continuously detects immediate physical hazards such as vehicles, persons, and obstacles. Stairs and drop-offs require a custom-trained hazard class set (see [Model Setup](#model-setup)).
* **Performance:** Runs completely offline, with no network dependency
* **Feedback:** Immediate local audio warnings (e.g., "Stop. Car.") via eSpeak when hazards are within proximity

### 2. Cognitive Vision (Intelligence Engine)
* **Technology:** OpenAI GPT-4o Vision + Whisper API
* **Function:** Scene description, verbatim document and label reading (OCR), currency identification, and barcode scanning (PyZbar)
* **Interaction:** Voice-activated, e.g. "Edith, what is in front of me?"
* **Streaming:** Sentence-level token streaming begins speech as soon as the first sentence is ready (~1.2s perceived response)

### 3. Offline Voice Activation
* **Technology:** Vosk Offline Speech Recognition (`vosk-model-small`)
* **Function:** Always-listening wake word detection (default: "Edith") that runs entirely on-device
* **Activation feedback:** A psychoacoustic activation chime (`chime.mp3`) plays locally the moment the wake word is heard
* **Dictation:** Continuous one-shot dictation captures the command that follows the wake word

### 4. Edge-Bypass Router
* Intercepts routine queries (time, battery level, system status, navigation toggles) and answers them locally in <50ms, saving latency, cost, and bandwidth.

### 5. Background Media Engine
* Headless, asynchronous YouTube audio playback via `yt-dlp` + `mpv` (~2.0s to start), running as a separate process so hazard detection is never blocked.
* "Edith, stop music" kills the playback process instantly.

### 6. Latency & Memory Optimizations
* **Speculative frame capture:** Frames are pre-encoded to a RAM-disk (`/dev/shm`) so the image is ready before the cloud request fires
* **Sentence-level token streaming:** Speech output starts before the full GPT-4o response is complete
* **zram-tools:** Compressed swap in RAM to stay stable on 512MB
* **Frame throttling:** The Reflex Engine processes a subset of frames to maintain system stability

---

## System Architecture
The software is multi-threaded to make the most of the Raspberry Pi Zero 2 W's limited resources:

| Thread | Responsibility |
|---|---|
| **Thread 0 (Dedicated Camera Bus)** | OpenCV V4L2 stream with automatic driver-recovery restart. Feeds frames to the rest of the system. |
| **Thread 1 (Reflex Safety Engine)** | Offline TensorFlow Lite hazard detection with local audio alerts (<50ms target response). |
| **Thread 2 (Voice & Cognitive Router)** | Vosk wake-word engine, local Edge-Bypass interceptor, speculative RAM-disk encoding (`/dev/shm`), and the streamed GPT-4o Vision pipeline (~1.2s response). |
| **Background Process (Media Engine)** | Asynchronous headless audio player using `yt-dlp` and `mpv`. |

```
                      ┌────────────────────────┐
                      │  Thread 0: Camera Bus   │
                      │  OpenCV V4L2 + auto-    │
                      │  restart recovery       │
                      └───────────┬────────────┘
                                  │ frames
                 ┌────────────────┴─────────────────┐
                 ▼                                   ▼
      ┌────────────────────┐             ┌──────────────────────────┐
      │ Thread 1: Reflex    │             │ /dev/shm (RAM-disk)       │
      │ Safety Engine       │             │ speculative frame encode  │
      │ TFLite (<50ms)      │             └────────────┬─────────────┘
      └─────────┬──────────┘                          │
                │ hazard                              ▼
                ▼                       ┌──────────────────────────────┐
      ┌────────────────────┐            │ Thread 2: Voice & Cognitive   │
      │ eSpeak (local TTS)  │◄───────────│ Router                        │
      │ "Stop. Car."        │  local     │  Mic ─► Vosk wake word        │
      └────────────────────┘  replies    │         ("Edith" ─► chime)    │
                                          │  ┌──────────┴───────────┐     │
                                          │  ▼                      ▼     │
                                          │ Edge-Bypass         Whisper ─► │
                                          │ Router (<50ms)      GPT-4o     │
                                          │ time/battery/       Vision     │
                                          │ status/toggles      (stream)   │
                                          │  │                      │     │
                                          │  ▼                      ▼     │
                                          │ eSpeak            mpg123 TTS  │
                                          └──────────────┬────────────────┘
                                                         │ "play [song]" /
                                                         │ "stop music"
                                                         ▼
                                          ┌──────────────────────────────┐
                                          │ Background Process: Media     │
                                          │ Engine (yt-dlp + mpv, headless)│
                                          └──────────────────────────────┘
```

---

## Hardware Requirements
* **SBC:** Raspberry Pi Zero 2 W (512MB RAM, optimized with zram-tools)
* **Camera:** Raspberry Pi Camera Module (v2 or v3)
* **Microphone:** USB Mini Microphone (via Micro-USB OTG adapter)
* **Audio Output:** 3.5mm Headphones or Bone Conduction Headset
* **Power Supply:** 5V 2.5A Power Bank (10,000mAh recommended)
* **Storage:** 16GB MicroSD Card (Class 10 or higher)

## Software Prerequisites
* Raspberry Pi OS (Legacy Bullseye recommended for camera compatibility)
* Python 3.9 or higher
* Internet connection (Wi-Fi or Mobile Hotspot) for GPT-4o, Whisper, and music streaming
* An [OpenAI API key](https://platform.openai.com/api-keys) with GPT-4o and Whisper access

> No wake-word account or access key is required. Wake-word detection runs fully offline with Vosk.

---

## Installation

### 1. System Update & Dependencies
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-pyaudio libatlas-base-dev portaudio19-dev
sudo apt-get install -y espeak mpg123 mpv zram-tools libzbar0
```

### 2. Configure ZRAM (Memory Compression)
The Pi Zero 2 W has only 512MB of RAM. Enable compressed swap to avoid out-of-memory crashes:
```bash
sudo nano /etc/default/zramswap
# Set, for example:
#   ALGO=lz4
#   PERCENT=50
sudo systemctl restart zramswap
swapon --show   # verify the zram device is active
```

### 3. Enable the Camera Interface
```bash
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
sudo reboot
```

### 4. Clone the Repository
```bash
git clone https://github.com/<your-username>/third-eye.git
cd third-eye
```

### 5. Create a Virtual Environment & Install Python Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Key Python libraries: `vosk`, `yt-dlp`, `psutil`, `pyzbar`, `opencv-python-headless`, plus the TFLite runtime and OpenAI client.

### 6. Configure Environment Variables
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
WAKE_WORD=edith
```

### 7. Download the Models
See [Model Setup](#model-setup) below for the TFLite hazard model and the Vosk speech model.

---

## Model Setup

### TFLite Hazard Model (Reflex Engine)
The Reflex Engine expects a quantized `SSD MobileNet V2` `.tflite` model at `reflex_engine/models/ssd_mobilenet_v2.tflite`. Pre-trained weights compatible with the TFLite runtime are available from the [TensorFlow model zoo](https://github.com/tensorflow/models). The stock COCO model does not include stairs or drop-offs; train and quantize a custom model for those hazard classes.

### Vosk Wake-Word Model
Download and extract `vosk-model-small-en-us-0.15` into the `models/` directory:
```bash
mkdir -p models && cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
rm vosk-model-small-en-us-0.15.zip
cd ..
```
The result should be `models/vosk-model-small-en-us-0.15/`.

---

## Usage

### Running the Application
```bash
source venv/bin/activate
python3 main.py
```

### Voice Commands
Say the wake word "Edith" first, then your command. Saying "Edith" alone plays the activation chime and listens for a one-shot command.

| Category | Command | Action | Typical Latency |
|---|---|---|---|
| Activation | "Edith" | Plays the local activation chime (`chime.mp3`) and listens | Instant |
| Edge-Bypass (local) | "Edith, what time is it?" | Local time readout, no cloud call | <50ms |
| Edge-Bypass (local) | "Edith, battery level" / "system status" | Local status readout | <50ms |
| Edge-Bypass (local) | Navigation toggles | Toggles handled locally | <50ms |
| One-shot Vision | "Edith, what is in front of me?" | Speculative RAM capture + GPT-4o Vision scene description | ~1.2s |
| One-shot Vision | "Edith, read this prescription label" | Speculative RAM capture + streamed GPT-4o Vision reading | ~1.2s |
| One-shot Vision | "Edith, what is this note?" | Currency identification | ~1.2s |
| Music | "Edith, play [song name]" | Spawns background `yt-dlp` + `mpv` audio process | ~2.0s |
| Music | "Edith, stop music" | Kills the background `mpv` process instantly | Instant |

*Latencies are typical figures on a Pi Zero 2 W with a good connection; cloud times vary with network quality.*

### Running on Boot
```bash
sudo cp systemd/third-eye.service /etc/systemd/system/
sudo systemctl enable third-eye.service
sudo systemctl start third-eye.service
```

---

## Project Structure
```
third-eye/
├── main.py                  # Entry point: spawns threads and background process
├── camera/
│   └── camera_bus.py        # Thread 0: OpenCV V4L2 stream + auto-restart
├── reflex_engine/
│   ├── detector.py          # Thread 1: TFLite inference + hazard logic
│   └── models/              # Quantized SSD MobileNet V2 (.tflite)
├── voice/
│   ├── wake_word.py         # Vosk offline wake-word listener
│   ├── edge_bypass.py       # Local router: time, battery, status, toggles
│   └── tts.py               # eSpeak (local) / mpg123 (cloud) output
├── intelligence_engine/
│   ├── vision_client.py     # GPT-4o Vision + Whisper, sentence-level streaming
│   └── prompts.py           # Prompt templates (scene, OCR, currency)
├── media/
│   └── player.py            # yt-dlp + mpv background audio process
├── assets/
│   └── chime.mp3            # Activation chime
├── models/                  # Vosk model (vosk-model-small-en-us-0.15)
├── systemd/
│   └── third-eye.service
├── requirements.txt
├── .env.example
└── README.md
```

---

## Roadmap

### Phase 1: Current Software MVP
- [x] Multi-threaded Python architecture (camera bus, reflex engine, voice and cloud router)
- [x] RAM-disk (`/dev/shm`) speculative frame pre-encoding
- [x] Edge-Bypass routing for local queries
- [x] Local TFLite hazard alerts
- [x] Background music streaming (`yt-dlp` + `mpv`)

### Phase 2: Post-Incubation Hardware Shift
- [ ] Bluetooth Hands-Free Profile (HFP) phone call handling and tethering
- [ ] Onboard MPU6050 accelerometer with SOS fall detection
- [ ] Migration to a custom rigid-flex PCB with an Edge AI SoC (ESP32-P4 / Rockchip RV1106)
- [ ] Port of the Reflex Engine to compiled C++ TinyML

## Troubleshooting
| Issue | Likely Cause | Fix |
|---|---|---|
| No camera feed | Camera interface disabled, or V4L2 driver stalled | Re-run `sudo raspi-config` and enable Camera; the camera thread auto-restarts on driver failure |
| Wake word not detected | Vosk model missing or in the wrong folder | Confirm `models/vosk-model-small-en-us-0.15/` exists |
| No audio output | Wrong output device selected | Run `sudo raspi-config` → Audio → select 3.5mm jack |
| High latency on GPT-4o calls | Weak Wi-Fi/hotspot signal | Move closer to router or use a dedicated mobile hotspot |
| Out-of-memory crashes or freezes | ZRAM not active | Check `swapon --show` and re-run the [ZRAM setup](#2-configure-zram-memory-compression) |
| Music won't start | `mpv`/`yt-dlp` missing or outdated | `sudo apt-get install mpv` and `pip install -U yt-dlp` |
| Barcode scan fails to import | `libzbar0` missing | `sudo apt-get install libzbar0` |

## Contributing
Contributions are welcome. Please open an issue to discuss significant changes before submitting a pull request.
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch and open a Pull Request

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
* [TensorFlow Lite](https://www.tensorflow.org/lite) for the on-device inference engine
* [Vosk](https://alphacephei.com/vosk/) for offline speech recognition
* [OpenAI GPT-4o and Whisper](https://openai.com/) for cognitive scene understanding and transcription
* [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [mpv](https://mpv.io/) for background audio streaming
