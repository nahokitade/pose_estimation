import datetime

class EventsPerSecond:
  def __init__(self, max_events=1000):
    self._start = None
    self._max_events = max_events
    self._timestamps = []

  def start(self):
    self._start = datetime.datetime.now().timestamp()

  def update(self):
    self._timestamps.append(datetime.datetime.now().timestamp())
    # truncate the list when it goes 100 over the max_size
    if len(self._timestamps) > self._max_events + 100:
      self._timestamps = self._timestamps[(1 - self._max_events):]

  def eps(self, last_n_seconds=10):
    # compute the (approximate) events in the last n seconds
    now = datetime.datetime.now().timestamp()
    seconds = min(now - self._start, last_n_seconds)
    return len([t for t in self._timestamps if t > (now - last_n_seconds)]) / seconds
