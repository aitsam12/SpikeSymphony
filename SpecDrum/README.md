# 🥁 SpecDrum: Vibration-Based Interactive Drumming with Event Cameras

**SpecDrum** is an interactive demo that transforms vibrational input into real-time drum beats using a neuromorphic vision sensor. Leveraging the high temporal resolution of event cameras, SpecDrum detects spatially localised frequency bursts and maps them to musical drum sounds across four vertical regions.

---

## 🥁 SpecDrum Demo Video

Scan the QR code below to watch the SpecDrum demonstration:

![QR Code – SpecDrum Video](https://api.qrserver.com/v1/create-qr-code/?data=https://drive.google.com/file/d/1cnroQZ1Y-kGFFjX0Xo7LP2N9fhJieFDf/view?usp=sharing&size=220x220)

---

## 🎯 Overview

In this demo, physical vibrations—such as tapping or striking sticks—are picked up by an event camera (e.g., SilkyEvCam). The sensor output is analysed for high-frequency activity across four regions of interest, triggering a beat if the energy and persistence surpass thresholds.

This system demonstrates how **FrequencyMapAsyncAlgorithm** from the Metavision SDK can be repurposed for high-speed musical interaction and intuitive audio feedback from spatially distributed input.

---

## 🧠 Key Features

- Region-wise frequency burst detection across 4 vertical columns
- Uniform beat length adapted to the shortest loaded sample
- Adjustable persistence threshold and event count for activation
- Real-time audio playback using `pygame.mixer` with polyphony support
- Live heatmap GUI showing spatial frequency intensity and triggering region
- Works with `.raw` recordings or live camera feed (e.g., SilkyEvCam)

---

## 🖼️ System Architecture

```text
Event Camera Input → EventsIterator
         ↓
[Polarity & Activity Filtering]
         ↓
[FrequencyMapAsyncAlgorithm]
         ↓
[ROI-Based Energy Accumulation]
         ↓
[Persistence + Event Count Check]
         ↓
[Drum Sample Triggering]


## 🧩 Requirements

Before running the demo, ensure the following are installed:

### 💻 Python Environment

- Python 3.8 or later  
- [Prophesee Metavision SDK](https://www.prophesee.ai/metavision-intelligence/) (with `core`, `cv`, `analytics`, and `ui` modules)

### 📦 Python Dependencies

Install via pip:

```bash
pip install numpy opencv-python pygame
