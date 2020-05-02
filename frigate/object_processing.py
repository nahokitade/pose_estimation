import hashlib
import cv2
import threading
import numpy as np
from collections import defaultdict
import pyarrow.plasma as plasma

EDGES = (
  ('nose', 'left eye'),
  ('nose', 'right eye'),
  ('nose', 'left ear'),
  ('nose', 'right ear'),
  ('left ear', 'left eye'),
  ('right ear', 'right eye'),
  ('left eye', 'right eye'),
  ('left shoulder', 'right shoulder'),
  ('left shoulder', 'left elbow'),
  ('left shoulder', 'left hip'),
  ('right shoulder', 'right elbow'),
  ('right shoulder', 'right hip'),
  ('left elbow', 'left wrist'),
  ('right elbow', 'right wrist'),
  ('left hip', 'right hip'),
  ('left hip', 'left knee'),
  ('right hip', 'right knee'),
  ('left knee', 'left ankle'),
  ('right knee', 'right ankle'),
)


class TrackedObjectProcessor(threading.Thread):
  def __init__(self, config, client, topic_prefix, tracked_objects_queue):
    threading.Thread.__init__(self)
    self.config = config
    self.client = client
    self.topic_prefix = topic_prefix
    self.tracked_objects_queue = tracked_objects_queue
    self.plasma_client = plasma.connect("/tmp/plasma")
    self.camera_data = defaultdict(lambda: {
      'object_status': defaultdict(lambda: defaultdict(lambda: 'OFF')),
      'tracked_objects': {},
      'current_frame': np.zeros((720, 1280, 3), np.uint8),
      'object_id': None
    })

  def get_current_frame(self, camera):
    return self.camera_data[camera]['current_frame']

  def draw_pose(self, frame, pose, threshold=0.2):
    xys = {}
    for label, keypoint in pose.keypoints.items():
      if keypoint.score < threshold: continue
      # Offset and scale to source coordinate space.
      kp_y = int(keypoint.yx[0])
      kp_x = int(keypoint.yx[1])

      xys[label] = (kp_x, kp_y)
      cv2.circle(frame, (int(kp_x), int(kp_y)), 5, (0, 0, 255), 2)

    for a, b in EDGES:
      if a not in xys or b not in xys: continue
      ax, ay = xys[a]
      bx, by = xys[b]
      cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)

  def run(self):
    while True:
      camera, frame_time, tracked_objects = self.tracked_objects_queue.get()

      config = self.config[camera]
      current_object_status = self.camera_data[camera]['object_status']
      self.camera_data[camera]['tracked_objects'] = tracked_objects

      ###
      # Draw tracked objects on the frame
      ###
      object_id_hash = hashlib.sha1(str.encode(f"{camera}{frame_time}"))
      object_id_bytes = object_id_hash.digest()
      object_id = plasma.ObjectID(object_id_bytes)
      current_frame = self.plasma_client.get(object_id, timeout_ms=0)

      if not current_frame is plasma.ObjectNotAvailable:
        # draw the poses on the frame
        for pose in tracked_objects.values():
         self.draw_pose(current_frame, pose)

        ###
        # Set the current frame as ready
        ###
        self.camera_data[camera]['current_frame'] = current_frame

        # store the object id, so you can delete it at the next loop
        previous_object_id = self.camera_data[camera]['object_id']
        if not previous_object_id is None:
          self.plasma_client.delete([previous_object_id])
        self.camera_data[camera]['object_id'] = object_id
      #
      # ###
      # # Report over MQTT
      # ###
      # # count objects with more than 2 entries in history by type
      # obj_counter = Counter()
      # for obj in tracked_objects.values():
      #   if len(obj['history']) > 1:
      #     obj_counter[obj['label']] += 1
      #
      # # report on detected objects
      # for obj_name, count in obj_counter.items():
      #   if new_status != current_object_status[obj_name]:
      #     self.client.publish(f"{self.topic_prefix}/{camera}/{obj_name}", new_status, retain=False)
      #
      # # expire any objects that are ON and no longer detected
      # expired_objects = [obj_name for obj_name, status in current_object_status.items() if
      #                    status == 'ON' and not obj_name in obj_counter]
      # for obj_name in expired_objects:
      #   current_object_status[obj_name] = 'OFF'
      #   self.client.publish(f"{self.topic_prefix}/{camera}/{obj_name}", 'OFF', retain=False)
      #   # send updated snapshot over mqtt
      #   best_frame = cv2.cvtColor(best_objects[obj_name]['frame'], cv2.COLOR_RGB2BGR)
      #   ret, jpg = cv2.imencode('.jpg', best_frame)
      #   if ret:
      #     jpg_bytes = jpg.tobytes()
      #     self.client.publish(f"{self.topic_prefix}/{camera}/{obj_name}/snapshot", jpg_bytes, retain=True)
