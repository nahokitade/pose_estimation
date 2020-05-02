import os
import time
import datetime
import cv2
import subprocess as sp
import numpy as np
import hashlib
import pyarrow.plasma as plasma
import SharedArray as sa
import json
from frigate.util import EventsPerSecond
from frigate.edgetpu import RemoteObjectDetector


def get_frame_shape(source):
  ffprobe_cmd = " ".join([
    'ffprobe',
    '-v',
    'panic',
    '-show_error',
    '-show_streams',
    '-of',
    'json',
    '"' + source + '"'
  ])
  print(ffprobe_cmd)
  p = sp.Popen(ffprobe_cmd, stdout=sp.PIPE, shell=True)
  (output, err) = p.communicate()
  p_status = p.wait()
  info = json.loads(output)
  print(info)

  video_info = [s for s in info['streams'] if s['codec_type'] == 'video'][0]

  if video_info['height'] != 0 and video_info['width'] != 0:
    return (video_info['height'], video_info['width'], 3)

  # fallback to using opencv if ffprobe didnt succeed
  video = cv2.VideoCapture(source)
  ret, frame = video.read()
  frame_shape = frame.shape
  video.release()
  return frame_shape


def get_ffmpeg_input(ffmpeg_input):
  frigate_vars = {k: v for k, v in os.environ.items() if k.startswith('FRIGATE_')}
  return ffmpeg_input.format(**frigate_vars)


def create_tensor_input(frame):
  resized_frame = frame
  if resized_frame.shape != (721, 1281, 3):
    resized_frame = cv2.resize(resized_frame, dsize=(721, 1281), interpolation=cv2.INTER_LINEAR)
  return resized_frame


def start_or_restart_ffmpeg(ffmpeg_cmd, frame_size, ffmpeg_process=None):
  if not ffmpeg_process is None:
    print("Terminating the existing ffmpeg process...")
    ffmpeg_process.terminate()
    try:
      print("Waiting for ffmpeg to exit gracefully...")
      ffmpeg_process.wait(timeout=30)
    except sp.TimeoutExpired:
      print("FFmpeg didnt exit. Force killing...")
      ffmpeg_process.kill()
      ffmpeg_process.wait()

  print("Creating ffmpeg process...")
  print(" ".join(ffmpeg_cmd))
  return sp.Popen(ffmpeg_cmd, stdout=sp.PIPE, bufsize=frame_size * 10)


def track_camera(name, config, ffmpeg_global_config, global_objects_config, detection_queue, detected_objects_queue,
                 fps, skipped_fps, detection_fps):
  print(f"Starting process for {name}: {os.getpid()}")

  # Merge the ffmpeg config with the global config
  ffmpeg = config.get('ffmpeg', {})
  ffmpeg_input = get_ffmpeg_input(ffmpeg['input'])
  ffmpeg_global_args = ffmpeg.get('global_args', ffmpeg_global_config['global_args'])
  ffmpeg_hwaccel_args = ffmpeg.get('hwaccel_args', ffmpeg_global_config['hwaccel_args'])
  ffmpeg_input_args = ffmpeg.get('input_args', ffmpeg_global_config['input_args'])
  ffmpeg_output_args = ffmpeg.get('output_args', ffmpeg_global_config['output_args'])
  ffmpeg_cmd = (['ffmpeg'] +
                ffmpeg_global_args +
                ffmpeg_hwaccel_args +
                ffmpeg_input_args +
                ['-i', ffmpeg_input] +
                ffmpeg_output_args +
                ['pipe:'])

  expected_fps = config['fps']
  take_frame = config.get('take_frame', 1)

  if 'width' in config and 'height' in config:
    frame_shape = (config['height'], config['width'], 3)
  else:
    frame_shape = get_frame_shape(ffmpeg_input)

  frame_size = frame_shape[0] * frame_shape[1] * frame_shape[2]

  try:
    sa.delete(name)
  except:
    pass

  frame = sa.create(name, shape=frame_shape, dtype=np.uint8)
  object_detector = RemoteObjectDetector(name, detection_queue)
  ffmpeg_process = start_or_restart_ffmpeg(ffmpeg_cmd, frame_size)

  plasma_client = plasma.connect("/tmp/plasma")
  frame_num = 0
  avg_wait = 0.0
  fps_tracker = EventsPerSecond()
  skipped_fps_tracker = EventsPerSecond()
  fps_tracker.start()
  skipped_fps_tracker.start()
  object_detector.fps.start()
  while True:
    start = datetime.datetime.now().timestamp()
    frame_bytes = ffmpeg_process.stdout.read(frame_size)
    duration = datetime.datetime.now().timestamp() - start
    avg_wait = (avg_wait * 99 + duration) / 100

    if not frame_bytes:
      rc = ffmpeg_process.poll()
      if rc is not None:
        print(f"{name}: ffmpeg_process exited unexpectedly with {rc}")
        ffmpeg_process = start_or_restart_ffmpeg(ffmpeg_cmd, frame_size, ffmpeg_process)
        time.sleep(10)
      else:
        print(f"{name}: ffmpeg_process is still running but didnt return any bytes")
      continue

    # limit frame rate
    frame_num += 1
    if (frame_num % take_frame) != 0:
      continue

    fps_tracker.update()
    fps.value = fps_tracker.eps()
    detection_fps.value = object_detector.fps.eps()

    frame_time = datetime.datetime.now().timestamp()

    # Store frame in numpy array
    frame[:] = (np
                .frombuffer(frame_bytes, np.uint8)
                .reshape(frame_shape))

    # skip object detection if we are below the min_fps and wait time is less than half the average
    if frame_num > 100 and fps.value < expected_fps - 1 and duration < 0.5 * avg_wait:
      skipped_fps_tracker.update()
      skipped_fps.value = skipped_fps_tracker.eps()
      continue

    skipped_fps.value = skipped_fps_tracker.eps()

    # pose
    tensor_input = create_tensor_input(frame)

    region_detections = object_detector.detect(tensor_input)

    # now that we have refined our detections, we need to track objects

    # put the frame in the plasma store
    object_id = hashlib.sha1(str.encode(f"{name}{frame_time}")).digest()
    plasma_client.put(frame, plasma.ObjectID(object_id))
    # add to the queue
    detected_objects_queue.put((name, frame_time, region_detections))

  print(f"{name}: exiting subprocess")
