# Third Eye: AI-Powered Smart Glasses for the Visually Impaired

![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%20Zero%202%20W-c51a4a)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20development-yellow)

## Project Overview
Third Eye is a low-cost, assistive wearable device designed to provide visual independence to the visually impaired. Built on the Raspberry Pi Zero 2 W, it employs a **Hybrid-Edge Architecture** that runs two independent AI systems simultaneously: an offline edge model for immediate safety and a cloud-based model for deep cognitive understanding.

The device replaces the traditional white cane with an active, voice-controlled system that can detect obstacles in real-time and describe the world in natural language.

---

## Key Features

### 1. Zero-Latency Safety (Reflex Engine)
* **Technology:** TensorFlow Lite (Quantized SSD MobileNet V2)
* **Function:** Detects immediate physical hazards such as cars, persons, stairs, and obstacles
* **Performance:** Operates completely offline with sub-50ms latency
* **Feedback:** Provides immediate audio warnings (e.g., "Stop. Car.") when hazards are within proximity

### 2. Cognitive Vision (Intelligence Engine)
* **Technology:** OpenAI GPT-4o
* **Function:** Analyzes complex scenes, reads text (OCR), and identifies currency
* **Interaction:** Activated via voice command, allowing the user to ask "What is in front of me?" or "Read this page"

### 3. Voice Activation
* **Technology:** Picovoice Porcupine
* **Function:** Always-listening wake word detection (default: "Edith") optimized for low-power hardware
* **Innovation:** Uses a custom audio resampling algorithm to process 48kHz hardware input into 16kHz AI-compatible streams without blocking the CPU

---

## System Architecture
The software is designed with a multi-threaded architecture to maximize the limited resources of the Raspberry Pi Zero 2 W:

| Thread | Responsibility |
|---|---|
| **Thread 1 (Vision)** | Handles camera capture and TFLite inference. Uses frame throttling (processing every 3rd frame) to maintain system stability. |
| **Thread 2 (Audio)** | Manages the microphone stream, wake-word detection, and audio output. |
| **Thread 3 (Network)** | Handles asynchronous API calls to OpenAI for image processing and Text-to-Speech (TTS) generation. |

```
                ┌─────────────────────┐
                │   Camera / Mic In    │
                └──────────┬───────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                                      ▼
┌───────────────┐                    ┌──────────────────┐
│ Thread 1       │                    │ Thread 2          │
│ Vision (TFLite)│                    │ Audio / Wake Word │
│ Reflex Engine  │                    │ (Porcupine)       │
└───────┬────────┘                    └─────────┬─────────┘
        │ hazard detected                        │ wake word / query
        ▼                                        ▼
┌───────────────┐                    ┌──────────────────┐
│ Local TTS      │                    │ Thread 3          │
│ "Stop. Car."   │                    │ Network (GPT-4o)  │
└───────────────┘                    │ Intelligence Engine│
                                       └─────────┬─────────┘
                                                 ▼
                                       ┌──────────────────┐
                                       │ TTS Output         │
                                       └──────────────────┘
```

---

## Hardware Requirements
* **SBC:** Raspberry Pi Zero 2 W
* **Camera:** Raspberry Pi Camera Module (v2 or v3)
* **Microphone:** USB Mini Microphone (via Micro-USB OTG adapter)
* **Audio Output:** 3.5mm Headphones or Bone Conduction Headset
* **Power Supply:** 5V 2.5A Power Bank (10,000mAh recommended)
* **Storage:** 16GB MicroSD Card (Class 10 or higher)

## Software Prerequisites
* Raspberry Pi OS (Legacy Bullseye recommended for camera compatibility)
* Python 3.9 or higher
* Internet connection (Wi-Fi or Mobile Hotspot) for GPT-4o features
* An [OpenAI API key](https://platform.openai.com/api-keys) with GPT-4o access
* A [Picovoice Access Key](https://console.picovoice.ai/) for wake-word detection

---

## Installation

### 1. System Update & Dependencies
Update the package list and install necessary system libraries:
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-pyaudio libatlas-base-dev portaudio19-dev
sudo apt-get install -y espeak mpg123
```

### 2. Enable the Camera Interface
```bash
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
sudo reboot
```

### 3. Clone the Repository
```bash
git clone https://github.com/<your-username>/third-eye.git
cd third-eye
```

### 4. Create a Virtual Environment & Install Python Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the project root with your API credentials:
```bash
OPENAI_API_KEY=your_openai_api_key_here
PICOVOICE_ACCESS_KEY=your_picovoice_access_key_here
WAKE_WORD=edith
```

### 6. Download the TFLite Model
Place the quantized SSD MobileNet V2 model in the `models/` directory (see [Model Setup](#model-setup) below).

---

## Usage

### Running the Application
```bash
source venv/bin/activate
python3 main.py
```

### Voice Commands
| Command | Action |
|---|---|
| "Edith" (wake word) | Activates the Intelligence Engine listener |
| "What is in front of me?" | Sends a snapshot to GPT-4o for scene description |
| "Read this page" | Runs OCR on the captured frame and reads it aloud |
| "What is this note?" | Identifies currency denomination |

### Running on Boot
To have Third Eye start automatically on power-up, enable the provided systemd service:
```bash
sudo cp systemd/third-eye.service /etc/systemd/system/
sudo systemctl enable third-eye.service
sudo systemctl start third-eye.service
```

---

## Project Structure
```
third-eye/
├── main.py                 # Entry point — spawns the three threads
├── reflex_engine/
│   ├── detector.py         # TFLite inference + hazard logic
│   └── models/              # Quantized SSD MobileNet V2 (.tflite)
├── intelligence_engine/
│   ├── vision_client.py    # GPT-4o API calls (scene, OCR, currency)
│   └── prompts.py          # Prompt templates for each query type
├── audio/
│   ├── wake_word.py        # Porcupine wake-word listener
│   ├── resampler.py        # 48kHz → 16kHz custom resampling
│   └── tts.py               # Text-to-speech output (espeak / mpg123)
├── systemd/
│   └── third-eye.service   # Boot-on-startup service definition
├── requirements.txt
├── .env.example
└── README.md
```

## Model Setup
The Reflex Engine expects a quantized `SSD MobileNet V2` `.tflite` model at `reflex_engine/models/ssd_mobilenet_v2.tflite`. Pre-trained weights compatible with the TFLite runtime are available from the [TensorFlow model zoo](https://github.com/tensorflow/models), or you may train/quantize a custom model for your target hazard classes.

---

## Roadmap
- [ ] GPS-based navigation and turn-by-turn audio guidance
- [ ] Bluetooth pairing for third-party bone-conduction headsets
- [ ] On-device caching of frequent GPT-4o responses to reduce latency
- [ ] Battery-level voice alerts
- [ ] Support for Raspberry Pi 4/5 for higher-throughput inference

## Troubleshooting
| Issue | Likely Cause | Fix |
|---|---|---|
| No camera feed | Camera interface disabled | Re-run `sudo raspi-config` and enable Camera |
| Wake word not detected | Invalid/missing Picovoice key | Check `.env` for `PICOVOICE_ACCESS_KEY` |
| No audio output | Wrong output device selected | Run `sudo raspi-config` → Audio → select 3.5mm jack |
| High latency on GPT-4o calls | Weak Wi-Fi/hotspot signal | Move closer to router or use a dedicated mobile hotspot |

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
* [OpenAI GPT-4o](https://openai.com/) for cognitive scene understanding
* [Picovoice Porcupine](https://picovoice.ai/) for efficient wake-word detection
