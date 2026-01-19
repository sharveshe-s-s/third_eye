# Third Eye: AI-Powered Smart Glasses for the Visually Impaired

## Project Overview
Third Eye is a low-cost, assistive wearable device designed to provide visual independence to the visually impaired. Built on the Raspberry Pi Zero 2 W, it employs a "Hybrid-Edge Architecture" that runs two independent AI systems simultaneously: an offline edge model for immediate safety and a cloud-based model for deep cognitive understanding.

The device replaces the traditional white cane with an active, voice-controlled system that can detect obstacles in real-time and describe the world in natural language.

## Key Features

### 1. Zero-Latency Safety (Reflex Engine)
* **Technology:** TensorFlow Lite (Quantized SSD MobileNet V2).
* **Function:** Detects immediate physical hazards such as cars, persons, stairs, and obstacles.
* **Performance:** Operates completely offline with sub-50ms latency.
* **Feedback:** Provides immediate audio warnings (e.g., "Stop. Car.") when hazards are within proximity.

### 2. Cognitive Vision (Intelligence Engine)
* **Technology:** OpenAI GPT-4o.
* **Function:** Analyzing complex scenes, reading text (OCR), and identifying currency.
* **Interaction:** Activated via voice command, allowing the user to ask "What is in front of me?" or "Read this page."

### 3. Voice Activation
* **Technology:** Picovoice Porcupine.
* **Function:** Always-listening wake word detection (default: "Edith") optimized for low-power hardware.
* **Innovation:** Uses a custom audio resampling algorithm to process 48kHz hardware input into 16kHz AI-compatible streams without blocking the CPU.

## System Architecture
The software is designed with a multi-threaded architecture to maximize the limited resources of the Raspberry Pi Zero 2 W:

* **Thread 1 (Vision):** Handles camera capture and TFLite inference. It uses frame throttling (processing every 3rd frame) to maintain system stability.
* **Thread 2 (Audio):** Manages the microphone stream, wake-word detection, and audio output.
* **Thread 3 (Network):** Handles asynchronous API calls to OpenAI for image processing and Text-to-Speech (TTS) generation.

## Hardware Requirements
* **SBC:** Raspberry Pi Zero 2 W
* **Camera:** Raspberry Pi Camera Module (v2 or v3)
* **Microphone:** USB Mini Microphone (via Micro-USB OTG adapter)
* **Audio Output:** 3.5mm Headphones or Bone Conduction Headset
* **Power Supply:** 5V 2.5A Power Bank (10,000mAh recommended)
* **Storage:** 16GB MicroSD Card (Class 10)

## Software Prerequisites
* Raspberry Pi OS (Legacy Bullseye recommended for camera compatibility)
* Python 3.9 or higher
* Internet connection (Wi-Fi or Mobile Hotspot) for GPT-4o features

## Installation

### 1. System Update & Dependencies
Update the package list and install necessary system libraries:
```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install python3-pip python3-pyaudio libatlas-base-dev portaudio19-dev
sudo apt-get install espeak mpg123

