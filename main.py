import numpy as np
import cv2
import matplotlib.pyplot as plt
import yolov5
import torch

# load pretrained model
model = yolov5.load('yolov5s.pt')
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
model.to(torch.device(0))  # i.e. device=torch.device(0)

cam = cv2.VideoCapture(0)

while True:
    
    result, image = cam.read()
    # Capture frame-by-frame 
    # inference with larger input size
    results = model(image)
    # parse results
    #predictions = results.pred[0]
    #boxes = predictions[:, :4] # x1, y1, x2, y2
    #scores = predictions[:, 4]
    #categories = predictions[:, 5]
    Final = np.squeeze(results.render())
    if result == True:
       
       cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE) 
       cv2.imshow('Frame',Final)

        # Press Q on keyboard to  exit
       if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    else: 
        break
 
# When everything done, release the video capture object
cam.release()
 
# Closes all the frames
cv2.destroyAllWindows()