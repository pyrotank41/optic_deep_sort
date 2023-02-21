import dataclasses
from utils import FPS, print_config
import cv2
import numpy as np
import os
import configparser
import pickle



root_path = '/Users/karansinghkochar/Documents/MOTdata/MOT17/train/MOT17-13-FRCNN/'
img_folder_path = os.path.join(root_path, 'img1/')
det_file_path = os.path.join(root_path, 'det/det.txt')
config_file = os.path.join(root_path, 'seqinfo.ini')

# get fps from config file
config = configparser.ConfigParser()
config.read(config_file)
print_config(config)
'''
name = MOT17-13-FRCNN
imdir = img1
framerate = 25
seqlength = 750
imwidth = 1920
imheight = 1080
imext = .jpg
'''

frame_rate = config['Sequence']['frameRate'] 
frame_delay = 1000 // int(frame_rate)
print(frame_delay)

# frame/sec 24frame/sec, sec/frame = 1/24 * 1000 = 41.6ms

files = os.listdir(img_folder_path)
files.sort()

fast = cv2.FastFeatureDetector_create()

def get_key_points(bboxs, img):
    ''' 
        input: bbox (x, y, w, h)
               img
        output: img
    '''
    image = img.copy()
    for bbox in bboxs:
        x, y, w, h = bbox[0].astype(np.int16)
        detection_class, confidance = bbox[2], bbox[1] 
        roi = image[y:y+h, x:x+w]
        keys = fast.detect(roi,None)
        # transform the key points to the original image
        keys_transformed = [cv2.KeyPoint(x=key.pt[0]+x, y= key.pt[1]+y, size=key.size, angle=key.angle, response=key.response, octave=key.octave, class_id=key.class_id) for key in keys]
        
        image = cv2.drawKeypoints(image, keys_transformed, None, color=(0,0,255))
    return image


def draw_bounding_box(bboxs, img, score=None, color=(0, 255, 0), thickness=2):
    ''' draws bounding box on the image
        input: bbox (detection_class, x, y, w, h, confidance)
               img
        output: img
    '''
    image = img.copy()
    
    for bbox in bboxs:
        x, y, w, h = bbox[0].astype(np.int16)
        detection_class, confidance = bbox[2], bbox[1]        
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        image = cv2.putText(image, str(confidance), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image

def get_saved_detections(file_path, gt=False):
    '''
        input: file_path: for detections.
        Detection's format for MOT: frame_number, class, x, y, w, h, score
        return: detections: dictionary where index is frame number and value is list 
        of detections at the frame number in format for ( [left,top,w,h], confidence, detection_class)
    '''
    
    if gt == True: 
        print("Ground_truth format is not implemented yet.")
        return
            
    detections = {}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            frame_number = int(line[0]) - 1
            
            if frame_number not in detections:
                detections[frame_number] = []
                
            # format for bb( [left,top,w,h], confidence, detection_class )
            tuple = ( np.array([line[2], line[3], line[4], line[5]]).astype(np.float16), float(line[6]), -1)
            detections[frame_number].append(tuple)
        return detections

# get detections        
detections = get_saved_detections(file_path=det_file_path)

# setup fps calculator for the video
fps = FPS()

from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=5)


def deep_sort_track(bbs,frame):
    # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
    tracks = tracker.update_tracks(bbs, frame=frame)
    bbs_ = []
    for track in tracks:
        track_id = track.track_id
        ltrb = track.to_ltwh()
        x, y, w, h = ltrb
        bbs_.append((np.array([x, y, w, h]), track_id, -1))
    img = draw_bounding_box(bbs_, frame)
    return img 

deepsort = True
start_frame = 150
for i, file_name in enumerate(files[start_frame:]):
    # open the image
    i = i + start_frame
    img = cv2.imread(img_folder_path + file_name)
    img = fps.draw_fps(img, frame_number=i+1)
    
    if deepsort:
        img = deep_sort_track(detections[i+1], img)
    else:
        img = draw_bounding_box(detections[i+1], img)
        img = get_key_points(detections[i+1], img)
    # show the image
    cv2.imshow('image', img)
    # wait for 1 ms
    cv2.waitKey(delay= 1)
