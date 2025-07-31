#!/usr/bin/env python3
"""
Vibration-based drum using event camera:
4 vertical regions (columns) with adjustable sensitivity
Uniform beat length equal to shortest sample
"""
import os
import argparse
import numpy as np
import cv2
import pygame
import threading

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PolarityFilterAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow

# --- Load Drum Samples & Determine Uniform Beat ---
drum_dir = '/home/aitsam/sounds_spiky_orchestra1/drums'
files = sorted(f for f in os.listdir(drum_dir)
               if f.lower().endswith(('.wav', '.mp3')))
if len(files) < 4:
    raise RuntimeError(f"Need at least 4 drum samples in {drum_dir}")

pygame.mixer.init()
drum_sounds = []
for f in files[:4]:
    snd = pygame.mixer.Sound(os.path.join(drum_dir, f))
    drum_sounds.append(snd)
# Uniform beat length: shortest duration
min_length_s = min(snd.get_length() for snd in drum_sounds)
BEAT_LENGTH_US = int(min_length_s * 1e6)
print(f"Uniform beat length: {min_length_s:.3f}s ({BEAT_LENGTH_US} µs)")

# --- Polyphony Manager ---
active_channels = []
MAX_VOICES = 32

def play_drum(idx):
    """Play drum sound for uniform beat length."""
    global active_channels
    snd = drum_sounds[idx]
    # Loop indefinitely and stop after uniform duration
    ch = snd.play(loops=-1)
    if ch:
        active_channels.append(ch)
        timer = threading.Timer(min_length_s + 0.01, lambda c=ch: c.stop())
        timer.daemon = True
        timer.start()
    # prune finished
    active_channels[:] = [c for c in active_channels if c.get_busy()]

# --- GUI Wrapper ---
class VibeGUI:
    def __init__(self, width, height):
        self.win = MTWindow(title='Vibration Drum', width=width, height=height,
                            mode=BaseWindow.RenderMode.BGR, open_directly=True)
    def show(self, frame):
        self.win.show_async(frame)
    def should_close(self):
        return self.win.should_close()
    def destroy(self):
        self.win.destroy()

# --- Arg Parsing ---
def parse_args():
    p = argparse.ArgumentParser(description='Vibration drum demo with uniform beat')
    p.add_argument('-i','--input-raw-file', dest='input_path', default="",
                   help='RAW file path or camera serial')
    p.add_argument('--no-display', action='store_true', help='Disable GUI')
    p.add_argument('--min-freq', type=float, default=1.0, help='Min frequency (Hz)')
    p.add_argument('--max-freq', type=float, default=500.0, help='Max frequency (Hz)')
    p.add_argument('--update-freq', dest='update_freq_hz', type=float, default=200.0,
                   help='Frequency algorithm update rate (Hz)')
    p.add_argument('-f','--replay-factor', dest='replay_factor', type=float, default=1.0,
                   help='Replay factor for RAW files')
    p.add_argument('--activity-ths', type=int, default=0,
                   help='Activity noise threshold (µs)')
    p.add_argument('--polarity', choices=('OFF','ON','ALL'), default='ALL',
                   help='Event polarity')
    p.add_argument('--diff-thresh', type=int, default=300,
                   help='Frequency diff threshold (µs)')
    p.add_argument('--min-events', type=int, default=5,
                   help='Minimum events in region to trigger')
    p.add_argument('--persist-frames', type=int, default=3,
                   help='Consecutive frames region must dominate')
    return p.parse_args()

# --- Main ---
def main():
    args = parse_args()

    # Event Iterator ~1 ms batches
    mv = EventsIterator(input_path=args.input_path,
                        start_ts=0, max_duration=None,
                        delta_t=1e3)
    if args.replay_factor > 0 and not is_live_camera(args.input_path):
        mv = LiveReplayEventsIterator(mv, replay_factor=args.replay_factor)
    height, width = mv.get_size()
    print(f"Event frame size: {width}x{height}")

    # Pre-filters
    filters = []
    if args.polarity != 'ALL':
        pol = 0 if args.polarity=='OFF' else 1
        filters.append(PolarityFilterAlgorithm(polarity=pol))
    if args.activity_ths > 0:
        filters.append(ActivityNoiseFilterAlgorithm(width=width,
                                                    height=height,
                                                    threshold=args.activity_ths))
    buf = filters[0].get_empty_output_buffer() if filters else None

    # Frequency Algorithm
    freq_algo = FrequencyMapAsyncAlgorithm(width=width,
                                           height=height,
                                           filter_length=1,
                                           min_freq=args.min_freq,
                                           max_freq=args.max_freq,
                                           diff_thresh_us=args.diff_thresh)
    freq_algo.update_frequency = args.update_freq_hz

    # Define 4 vertical regions
    region_w = width // 4
    num_regions = 4
    last_trigger = [0]*num_regions
    last_idx = -1
    persist_count = 0

    gui = None
    if not args.no_display:
        gui = VibeGUI(width, height)

    # Frequency callback
    def on_freq_map(ts, freq_map):
        nonlocal last_idx, persist_count
        counts = []
        for i in range(num_regions):
            x0 = i*region_w
            x1 = width if i==num_regions-1 else (i+1)*region_w
            roi = freq_map[:, x0:x1]
            n = int(np.count_nonzero(roi))
            counts.append(n)
        idx = int(np.argmax(counts))
        # Persistence logic
        if idx == last_idx:
            persist_count += 1
        else:
            persist_count = 1
            last_idx = idx
        # Trigger condition
        if (persist_count >= args.persist_frames and
            counts[idx] >= args.min_events and
            ts - last_trigger[idx] > BEAT_LENGTH_US):
            play_drum(idx)
            last_trigger[idx] = ts
            persist_count = 0
        # Display
        if gui:
            norm = cv2.normalize(freq_map, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
            heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            for j in range(1, num_regions):
                cv2.line(heat, (j*region_w,0), (j*region_w,height-1), (0,255,0), 2)
            gui.show(heat)

    freq_algo.set_output_callback(on_freq_map)

    # Main loop
    for evs in mv:
        EventLoop.poll_and_dispatch()
        if filters:
            filters[0].process_events(evs, buf)
            for f in filters[1:]: f.process_events_(buf)
            freq_algo.process_events(buf)
        else:
            freq_algo.process_events(evs)
        if gui and gui.should_close():
            break

    if gui:
        gui.destroy()

if __name__ == '__main__':
    main()
