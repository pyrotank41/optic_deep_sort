import time
import cv2

class FPS():
  def __init__(self):
    self.frame_count = 0
    self.time_start = time.time()
    self.fps = 0
  
  def update_fps(self):
    self.fps = self.get_fps()
      
  def get_fps(self):
    self.frame_count += 1
    time_end = time.time()
    time_diff = time_end - self.time_start
    self.fps = self.frame_count / time_diff
    if time_diff > 1:
        self.frame_count = 0
        self.time_start = time.time()
    return self.fps
  
  def draw_fps(self, frame, frame_number:int=None):
    self.update_fps()
    if frame_number is not None: frame = cv2.putText(frame, "Frame: " + str(frame_number) +  " FPS: " + str(int(self.fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else: frame = cv2.putText(frame, "FPS: " + str(int(self.fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame 
  
def print_config(config):
  for section in config.sections():
      print(section)
      for key in config[section]:
          print(key, config[section][key])