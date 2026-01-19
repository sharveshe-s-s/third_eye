Third Eye: Hybrid-Edge AI Smart Glasses for the Visually Impaired
Project Overview
Third Eye is an affordable, assistive wearable device designed to act as a digital visual cortex for the visually impaired. Built on the Raspberry Pi Zero 2 W, it utilizes a proprietary Hybrid-Edge Architecture to balance immediate physical safety with high-level cognitive understanding.

Unlike existing market solutions that rely entirely on the cloud or expensive hardware, Third Eye runs two independent AI engines simultaneously on a low-power microcomputer:

Reflex Engine (Edge AI): A quantized TensorFlow Lite model runs offline to detect immediate hazards (cars, stairs, people) in under 50 milliseconds.

Cognitive Engine (Cloud AI): OpenAI GPT-4o provides detailed scene description, text reading, and currency identification upon voice command.

Key Features
Zero-Latency Safety: Offline object detection ensures the user is warned of obstacles even without an internet connection.

Generative Vision: Uses GPT-4o to describe complex scenes, read vernacular text (Tamil/English), and identify objects in detail.

Voice-Activated Interface: Hands-free operation using Picovoice Porcupine wake-word detection ("Edith" or "Jarvis").

Resource Optimized: Runs flawlessly on 512MB RAM using custom audio resampling algorithms and multi-threaded architecture.

Cost Effective: Total hardware cost is under INR 8,000, significantly lower than commercial alternatives.

Hardware Requirements
SBC: Raspberry Pi Zero 2 W (Headless setup)

Camera: Raspberry Pi Camera Module 2 (or compatible CSI camera)

Audio Input: Mini USB Microphone (plugged via OTG adapter)

Audio Output: 3.5mm Jack Headphones or Bone Conduction Transducer

Power: 5V 2.5A Power Bank (10,000mAh recommended)

Connectivity: Mobile Hotspot or Wi-Fi

Software Architecture
The system consists of three main threads running in parallel:

Vision Thread: Captures video frames and performs SSD MobileNet V2 inference (Quantized) for object detection. It prioritizes specific hazard classes (e.g., vehicles, holes, obstacles).

Audio Loop (Main Thread): Listens for the wake word using a 16kHz resampled stream. Upon activation, it triggers the Scene Description module.

Network Handler: Manages secure API calls to OpenAI for image analysis and text-to-speech generation.

Installation and Setup
1. System Prerequisites
Ensure your Raspberry Pi is running Raspberry Pi OS (Bullseye or later) and system packages are updated.

Bash

sudo apt-get update && sudo apt-get upgrade
sudo apt-get install python3-pip python3-pyaudio libatlas-base-dev portaudio19-dev
sudo apt-get install espeak mpg123 libcamera-apps
2. Clone the Repository
Bash

git clone https://github.com/your-username/third-eye.git
cd third-eye
3. Install Python Dependencies
Bash

pip3 install -r requirements.txt
Note: If using a virtual environment, activate it before installing.

4. Configuration
Open final.py and update the following configuration variables:

API Keys: Replace the placeholder strings with your valid keys.

OPENAI_KEY: Your OpenAI API Key.

PICOVOICE_KEY: Your Picovoice Access Key.

Model Paths: Ensure the path to your .tflite model and .ppn wake word file is correct in the script.

Python

KEYWORD_FILE_PATH = "/home/admin/models/edith.ppn"
MODEL_PATH = "/home/admin/models/fast.tflite"
Usage
1. Hardware Calibration
Before running the main script, ensure the microphone is active and unmuted.

Bash

alsamixer
(Press F6 to select the USB Audio device and set Capture volume to 80-90%).

2. Running the System
Execute the main application script:

Bash

python3 final.py
3. Operation
Hazard Detection: The system automatically speaks warnings (e.g., "Stop. Car.") when a hazardous object is detected within a close proximity range.

Scene Description: Say the wake word (default: "Edith"). The system will capture an image, analyze it using GPT-4o, and read aloud a description of the scene.

Troubleshooting
Audio Input Failed
If the system crashes with "Input Overflow" or cannot find the microphone:

Run sudo fuser -k /dev/snd/* to kill background audio processes.

Verify the microphone index in the audio_loop function (Hardcode mic_index = 0 if using a single USB mic).

Latency Issues
If the audio stutters:

Ensure the Power Supply provides stable voltage.

The script includes a frame throttling mechanism (processing every 3rd frame). Do not remove this, as the Pi Zero 2 W cannot handle 30FPS inference alongside Audio processing.

Acknowledgements
TensorFlow Lite for the object detection framework.

OpenAI for the GPT-4o and TTS APIs.

Picovoice for the Porcupine wake word engine.
