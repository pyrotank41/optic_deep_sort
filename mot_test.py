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
        x, y, w, h = bbox[1:-1].astype(np.int32)
        detection_class, confidance = bbox[0], bbox[-1]
        roi = image[y:y+h, x:x+w]
        keys = fast.detect(roi,None)
        # transform the key points to the original image
        keys_transformed = [cv2.KeyPoint(x=key.pt[0]+x, y= key.pt[1]+y, size=key.size, angle=key.angle, response=key.response, octave=key.octave, class_id=key.class_id) for key in keys]
        
        image = cv2.drawKeypoints(image, keys_transformed, None, color=(0,0,255))
    return image, keys_transformed, keys


def draw_bounding_box(bboxs, img, score=None, color=(0, 255, 0), thickness=2):
    ''' draws bounding box on the image
        input: bbox (x, y, w, h)
               img
        output: img
    '''
    image = img.copy()
    
    for bbox in bboxs:
        x, y, w, h = bbox[1:-1].astype(np.int32)
        detection_class, confidance = bbox[0], bbox[-1]        
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        image = cv2.putText(image, str(confidance), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image

def get_saved_detections(file_path):
    '''
        input: file_path: for detections.
        Detection's format for MOT: frame_number, class, x, y, w, h, score
        return: detections: dictionary where index is frame number and value is list 
        of detections at the frame number in the format of class, x, y, w, h, score
    '''
    detections = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            frame_number = int(line[0])
            if frame_number not in detections:
                detections[frame_number] = []
            # class, x, y, w, h, score
            detections[frame_number].append(np.array([line[1], line[2], line[3], line[4], line[5], line[6]]).astype(np.float32))
        return detections

# get detections        
detections = get_saved_detections(file_path=det_file_path)

# setup fps calculator for the video
fps = FPS()

for i, file_name in enumerate(files):
    # open the image
    img = cv2.imread(img_folder_path + file_name)
    img = fps.draw_fps(img)
    img = draw_bounding_box(detections[i+1], img, score=0.9)
    img = get_key_points(detections[i+1], img)
    # show the image
    cv2.imshow('image', img)
    # wait for 1 ms
    
    cv2.waitKey(delay= 1)
