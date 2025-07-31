# 🎹 Spiky Piano: Size-Based Event-Driven Musical Interaction

**Spiky Piano** is an interactive, real-time demo that uses an event-based camera to detect object sizes and trigger corresponding musical notes. Developed using Prophesee’s neuromorphic vision SDK, this project showcases how spatial blob analysis on asynchronous data can drive high-speed, low-latency multimedia applications.

---

## 🎯 Overview

Objects of varying sizes are thrown or moved in the field of view of an event camera. The system dynamically classifies the object size into **small**, **medium**, or **large**, and triggers a musical note from the corresponding sound bank.

The demo exemplifies:
- **Low-latency size categorisation**
- **Real-time audio triggering**
- **Intuitive visual GUI with calibration bars**
- **Neuromorphic visual processing pipeline**

---

## 🧠 Key Features

- Event stream pre-processing via morphological closing and downsampling
- Size estimation from connected component area on binary event frames
- Adjustable thresholds for small and medium blob areas
- Support for both **live camera input** and **offline `.raw` file replay**
- Interactive GUI: live stats, thresholds, reset button, and visual overlay
- Real-time polyphonic audio playback with fading grains
- Configurable lines for vertical event counting

---

## 🖼️ System Architecture

```text
Event Camera Input → Event Iterator
     ↓
[Polarity & Noise Filtering]
     ↓
[Downsampled Event Frame + Morph Closing]
     ↓
[Connected Component Analysis]
     ↓
[Area-based Classification: Small / Medium / Large]
     ↓
[Sound Trigger via PyGame]


## Installation Requirements

Python 3.8+

Prophesee Metavision SDK 4.x or later (Core, CV, Analytics modules)

Dependencies:

pip install numpy opencv-python pygame


## Audio Samples

spiky_piano/
├── nodes/
│   ├── small/
│   ├── medium/
│   └── large/


## Running the Demo

cd spiky_piano

python3 spikypiano_60fps.py -i YOUR_INPUT.raw
