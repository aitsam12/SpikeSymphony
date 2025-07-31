#!/usr/bin/env python3
"""
Size-categorised event-camera piano demo (Optimised):
- Spatial downsampling (½ resolution) for blob detection (×4 speed-up)
- Morphological closing on downsampled image
- Blob area from CC_STAT_AREA, scaled back to full-resolution
- Dynamic thresholds: small <1%, medium <10% of frame area
- No note lasts more than 50 ms (maxtime=50)
- Enhanced display: calibration bars & reset button
"""
import os
import random
import sys
import time
import pygame
import numpy as np
import cv2
import argparse

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PolarityFilterAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TransposeEventsAlgorithm
from metavision_sdk_analytics import CountingAlgorithm, CountingCalibration

# Calibration bar and button config
BAR_HEIGHT = 10  # height of calibration bars
BUTTON_TEXT = 'Reset'
BUTTON_FONT = cv2.FONT_HERSHEY_SIMPLEX
BUTTON_SCALE = 0.5
BUTTON_THICK = 1
BUTTON_PADDING = 5

reset_requested = False

def on_mouse(event, x, y, flags, param):
    global reset_requested
    btn_x, btn_y, btn_w, btn_h = param['btn_rect']
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_x <= x <= btn_x + btn_w and btn_y <= y <= btn_y + btn_h:
            reset_requested = True


def make_grain(sound: pygame.mixer.Sound, release_ms: int = 20) -> pygame.mixer.Sound:
    freq, _, _ = pygame.mixer.get_init()
    arr = pygame.sndarray.array(sound)
    fade_samples = int(freq * (release_ms / 1000.0))
    if fade_samples <= 0 or fade_samples >= arr.shape[0]:
        return sound
    window = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    if arr.ndim == 1:
        arr[-fade_samples:] = (arr[-fade_samples:].astype(np.float32) * window).astype(arr.dtype)
    else:
        for ch in range(arr.shape[1]):
            arr[-fade_samples:, ch] = (arr[-fade_samples:, ch].astype(np.float32) * window).astype(arr.dtype)
    return pygame.sndarray.make_sound(arr)


def parse_args():
    parser = argparse.ArgumentParser(description='Size-categorised event-camera piano')
    parser.add_argument('-i','--input-raw-file', dest='input_path', default="",
                        help='RAW file or camera serial')
    parser.add_argument('--process-from', dest='process_from', type=int, default=0,
                        help='Start time (µs)')
    parser.add_argument('--process-to', dest='process_to', type=int, default=None,
                        help='End time (µs)')
    parser.add_argument('-f','--replay-factor', dest='replay_factor', type=float, default=1.0,
                        help='Replay factor (>1 slow, <1 fast)')
    filt = parser.add_argument_group('Filtering options')
    filt.add_argument('--activity-ths', dest='activity_ths', type=int, default=0,
                      help='Activity noise threshold (µs)')
    filt.add_argument('--polarity', dest='polarity', choices=('OFF','ON','ALL'), default='ON',
                      help='Event polarity to process')
    filt.add_argument('-r','--rotate', dest='rotate', action='store_true',
                      help='Rotate camera view 90°')
    algo = parser.add_argument_group('Algorithm options')
    algo.add_argument('-n','--num-lines', dest='num_lines', type=int, default=2,
                      help='Number of counting lines')
    algo.add_argument('--min-y', dest='min_y_line', type=int, default=250,
                      help='First line Y coordinate')
    algo.add_argument('--max-y', dest='max_y_line', type=int, default=250,
                      help='Last line Y coordinate')
    out = parser.add_argument_group('Outcome options')
    out.add_argument('--no-display', dest='no_display', action='store_true',
                     help='Disable OpenCV GUI')
    return parser.parse_args()


def main():
    global reset_requested
    args = parse_args()

    # Initialise audio
    pygame.mixer.init()
    base_dir = '/home/aitsam/sounds_spiky_orchestra1/nodes'
    def load_category(name):
        path = os.path.join(base_dir, name)
        sounds = []
        if os.path.isdir(path):
            for fn in sorted(os.listdir(path)):
                if fn.lower().endswith(('.mp3','.wav')):
                    snd = pygame.mixer.Sound(os.path.join(path, fn))
                    sounds.append(make_grain(snd, release_ms=20))
        else:
            print(f"Warning: folder not found {path}", file=sys.stderr)
        return sounds

    small_sounds  = load_category('small')
    medium_sounds = load_category('medium')
    large_sounds  = load_category('large')
    print(f"Loaded: small={len(small_sounds)}, medium={len(medium_sounds)}, large={len(large_sounds)}", file=sys.stderr)

    # Polyphony manager
    MAX_VOICES = 64
    active_channels = []
    def play_sound(snd: pygame.mixer.Sound):
        nonlocal active_channels
        if len(active_channels) >= MAX_VOICES:
            ch = active_channels.pop(0)
            if ch: ch.stop()
        snd.set_volume(1.0)
        ch = snd.play(loops=0, maxtime=50)
        if ch: active_channels.append(ch)

    # Event iterator
    mv_it = EventsIterator(input_path=args.input_path,
                           start_ts=args.process_from,
                           max_duration=(args.process_to - args.process_from) if args.process_to else None,
                           delta_t=1e3)
    if args.replay_factor > 0 and not is_live_camera(args.input_path):
        mv_it = LiveReplayEventsIterator(mv_it, replay_factor=args.replay_factor)

    height, width = mv_it.get_size()
    img_area = height * width

    # Prepare downsampling sizes
    w2, h2 = width // 2, height // 2
    small_thresh = img_area * 0.00075
    med_thresh   = img_area * 0.02
    # thresholds on downsampled pixel areas
    small_thresh_ds = small_thresh / 4.0
    med_thresh_ds   = med_thresh   / 4.0

    # Filters
    filters = []
    if args.polarity != 'ALL':
        filters.append(PolarityFilterAlgorithm(polarity=0 if args.polarity=='OFF' else 1))
    if args.rotate:
        filters.append(TransposeEventsAlgorithm())
        height, width = width, height
    if args.activity_ths > 0:
        filters.append(ActivityNoiseFilterAlgorithm(width=width, height=height, threshold=args.activity_ths))
    events_buf = filters[0].get_empty_output_buffer() if filters else None

    # Counting setup
    thr, acc = CountingCalibration.calibrate(width=width, height=height,
                                            object_min_size=0.7,
                                            object_average_speed=7.2,
                                            distance_object_camera=320.0)
    ca = CountingAlgorithm(width=width, height=height, cluster_ths=thr, accumulation_time_us=acc)
    ystep = (args.max_y_line - args.min_y_line) // (args.num_lines - 1)
    rows = [args.min_y_line + i*ystep for i in range(args.num_lines)]
    ca.add_line_counters(rows)
    current_count, last_count = 0, -1
    counts = {'small':0, 'medium':0, 'large':0}
    def counting_cb(ts, global_count, last_ts, _):
        nonlocal current_count
        current_count = global_count
    ca.set_output_callback(counting_cb)

    # Display setup
    if not args.no_display:
        cv2.namedWindow('Events', cv2.WINDOW_NORMAL)
        (tw, th), _ = cv2.getTextSize(BUTTON_TEXT, BUTTON_FONT, BUTTON_SCALE, BUTTON_THICK)
        btn_w, btn_h = tw + 2*BUTTON_PADDING, th + 2*BUTTON_PADDING
        btn_x, btn_y = width//2 - btn_w - 10, 10
        btn_rect = (btn_x, btn_y, btn_w, btn_h)
        cv2.setMouseCallback('Events', on_mouse, {'btn_rect':btn_rect})

    event_img = np.zeros((height, width), dtype=np.uint8)
    draw_interval = 1.0/60.0
    last_draw = time.perf_counter()

    # Structuring element for morphological closing
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for evs in mv_it:
        # Preprocess events
        if filters:
            filters[0].process_events(evs, events_buf)
            for f in filters[1:]: f.process_events_(events_buf)
            data = events_buf.numpy(copy=False)
        else:
            data = evs.numpy(copy=False)

        # Draw events
        event_img.fill(0)
        if data.size:
            ys = np.clip(data['y'], 0, height-1)
            xs = np.clip(data['x'], 0, width-1)
            event_img[ys, xs] = 255

        # Downsample for detection
        img_ds = cv2.resize(event_img, (w2, h2), interpolation=cv2.INTER_NEAREST)
        # Merge sparse events
        img_closed = cv2.morphologyEx(img_ds, cv2.MORPH_CLOSE, se)

        # Counting events crossing lines
        ca.process_events(events_buf if filters else evs)

        # Connected components on downsampled image
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(img_closed, connectivity=8)
        if num_labels > 1:
            comp_areas = stats[1:, cv2.CC_STAT_AREA]
            max_area_ds = float(comp_areas.max())
        else:
            max_area_ds = 0.0

        # Classify size (scale back to full-res via thresholds)
        if current_count != last_count:
            if max_area_ds < small_thresh_ds:
                cat, snd_list = 'small', small_sounds
            elif max_area_ds < med_thresh_ds:
                cat, snd_list = 'medium', medium_sounds
            else:
                cat, snd_list = 'large', large_sounds

            counts[cat] += 1  # Only increment the relevant category
            if snd_list:
                play_sound(random.choice(snd_list))
            last_count = current_count

        # Reset stats
        if reset_requested:
            counts = {'small':0, 'medium':0, 'large':0}
            reset_requested = False

        # Render display at 60 FPS
        now = time.perf_counter()
        if not args.no_display and (now - last_draw) >= draw_interval:
            vis = cv2.cvtColor(cv2.resize(img_closed, (width//2, height//2), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            # Counting lines
            for y in rows:
                cv2.line(vis, (0, y//2), (width//2-1, y//2), (0,255,0), 1)
            # Calibration bars
            sw = int((small_thresh_ds/(w2*h2))*(w2))
            mw = int((med_thresh_ds/(w2*h2))*(w2))
            cv2.rectangle(vis, (0, vis.shape[0]-BAR_HEIGHT), (sw, vis.shape[0]), (255,0,0), -1)
            cv2.rectangle(vis, (sw, vis.shape[0]-BAR_HEIGHT), (mw, vis.shape[0]), (0,255,255), -1)
            # Button
            bx,by,bw,bh = btn_rect
            cv2.rectangle(vis, (bx,by), (bx+bw,by+bh), (0,0,255), 1)
            cv2.putText(vis, BUTTON_TEXT, (bx+BUTTON_PADDING, by+bh-BUTTON_PADDING), BUTTON_FONT, BUTTON_SCALE, (0,0,255), BUTTON_THICK)
            # Stats text
            cv2.putText(vis, f"C={current_count}", (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.putText(vis, f"S={counts['small']} M={counts['medium']} L={counts['large']}", (10,40), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.putText(vis, f"Area_ds={int(max_area_ds)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.imshow('Events', vis)
            if cv2.waitKey(1) & 0xFF == 27: break
            last_draw = now

    if not args.no_display: cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
