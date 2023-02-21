import numpy as np
import scipy
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import time
from utils import FPS


if __name__ == "__main__":

  capture = cv2.VideoCapture(1)
  # display the webcam feed
  fps = FPS()
  fast = cv2.FastFeatureDetector_create()
  
  while True: 
    ret, frame = capture.read()
    keys = fast.detect(frame,None)
    frame = cv2.drawKeypoints(frame, keys, None, color=(0,0,255))

    # write fps on frame
    frame = fps.draw_fps(frame)
    cv2.imshow('frame', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break