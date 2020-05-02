import os
import datetime
import hashlib
import multiprocessing as mp
import pyarrow.plasma as plasma
from frigate.util import EventsPerSecond
from pose_engine import PoseEngine


class ObjectDetector:
  def __init__(self):
    try:
      self.engine = PoseEngine('/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite')

    except ValueError:
      print("No EdgeTPU detected. Failing.")

  def estimate_raw(self, tensor_input):
    poses, inference_time = self.engine.DetectPosesInImage(tensor_input)
    print('Inference time: %.fms' % inference_time)
    return poses


def run_detector(detection_queue, avg_speed, start):
  print(f"Starting detection process: {os.getpid()}")
  plasma_client = plasma.connect("/tmp/plasma")
  object_detector = ObjectDetector()

  while True:
    object_id_str = detection_queue.get()
    object_id_hash = hashlib.sha1(str.encode(object_id_str))
    object_id = plasma.ObjectID(object_id_hash.digest())
    object_id_out = plasma.ObjectID(hashlib.sha1(str.encode(f"out-{object_id_str}")).digest())
    input_frame = plasma_client.get(object_id, timeout_ms=0)

    if input_frame is plasma.ObjectNotAvailable:
      continue

    # detect and put the output in the plasma store
    start.value = datetime.datetime.now().timestamp()
    plasma_client.put(object_detector.estimate_raw(input_frame), object_id_out)
    duration = datetime.datetime.now().timestamp() - start.value
    start.value = 0.0

    avg_speed.value = (avg_speed.value * 9 + duration) / 10


class EdgeTPUProcess:
  def __init__(self):
    self.detection_queue = mp.Queue()
    self.avg_inference_speed = mp.Value('d', 0.01)
    self.detection_start = mp.Value('d', 0.0)
    self.detect_process = None
    self.start_or_restart()

  def start_or_restart(self):
    self.detection_start.value = 0.0
    if (not self.detect_process is None) and self.detect_process.is_alive():
      self.detect_process.terminate()
      print("Waiting for detection process to exit gracefully...")
      self.detect_process.join(timeout=30)
      if self.detect_process.exitcode is None:
        print("Detection process didnt exit. Force killing...")
        self.detect_process.kill()
        self.detect_process.join()
    self.detect_process = mp.Process(target=run_detector,
                                     args=(self.detection_queue, self.avg_inference_speed, self.detection_start))
    self.detect_process.daemon = True
    self.detect_process.start()


class RemoteObjectDetector:
  def __init__(self, name, detection_queue):
    self.name = name
    self.fps = EventsPerSecond()
    self.plasma_client = plasma.connect("/tmp/plasma")
    self.detection_queue = detection_queue

  def detect(self, tensor_input):
    detections = []

    now = f"{self.name}-{str(datetime.datetime.now().timestamp())}"
    object_id_frame = plasma.ObjectID(hashlib.sha1(str.encode(now)).digest())
    object_id_detections = plasma.ObjectID(hashlib.sha1(str.encode(f"out-{now}")).digest())
    self.plasma_client.put(tensor_input, object_id_frame)
    self.detection_queue.put(now)
    raw_detections = self.plasma_client.get(object_id_detections, timeout_ms=10000)

    if raw_detections is plasma.ObjectNotAvailable:
      self.plasma_client.delete([object_id_frame])
      return detections

    self.plasma_client.delete([object_id_frame, object_id_detections])
    self.fps.update()
    return raw_detections
