import numpy as np
import cv2
import matplotlib.pyplot as plt
import yolov5
import torch

# load pretrained model
model = yolov5.load('yolov5m.pt')
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
model.cpu  # i.e. device=torch.device(0)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

People = []
PeopleAmountInit = 0
lastImgs = []
Ticks = 0

class Box:
    def __init__(self,x1,y1,x2,y2):
        self.Pt1 = [float(x1),float(y1)]
        self.Pt2 = [float(x2),float(y2)]
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = y2

    def Amplify(self, Amount):
        x1 = max(0,self.x1 - int(Amount))
        y1 = max(0,self.y1 - int(Amount))
        x2 = self.x2 + int(Amount)
        y2 = self.y2 + int(Amount)
        self.__init__(x1,y1,x2,y2)
    def __str__(self):
        Message = "This box is made by the Points " + str(self.Pt1) + " and " + str(self.Pt2)
        return str(Message)
      

class Person:
   def __init__(self,name,box):
      self.box = box
      self.name = name
   def __str__(self):
       Message = str(self.name) + " is located in the box made of the points " + str(self.box.Pt1) + " and " + str(self.box.Pt2)
       return str(Message)
   
def Capture():
   lastImgs = []
   Ticks = 0   
   result, image = cam.read()
   results = model(image,size=1280)
   Predictions = results.xyxy[0] # 0 because it is image 1
   NewPredictions = []
   for Prediction in Predictions:
        if Prediction[-1] == 0:
            NewPredictions.append(Prediction)

   Count = 0

   for Prediction in NewPredictions:
       box = Box(Prediction[0],Prediction[1],Prediction[2],Prediction[3])
       box.Amplify(10)
       Count += 1
       People.append(Person("Joe " + str(Count),box))


Capture()
while True:
    Ticks += 1
    result, image = cam.read()
    images = []
    heatmaps = []

    BGsub = cv2.bgsegm.createBackgroundSubtractorMOG()
    for Pers in People:
        images.append(image[int(Pers.box.y1):int(Pers.box.y2),int(Pers.box.x1):int(Pers.box.x2)])

    i = 0
    if Ticks > 1:
        for img in images:
            grisNow = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fgmaskNow = BGsub.apply(grisNow)
            grisNow = cv2.GaussianBlur(fgmaskNow, (5, 5), 0)

            grisBefore = cv2.cvtColor(lastImgs[i], cv2.COLOR_BGR2GRAY)
            fgmaskBefore = BGsub.apply(grisBefore)
            grisBefore = cv2.GaussianBlur(fgmaskBefore, (5, 5), 0)

            restados = cv2.absdiff(grisBefore, grisNow)

            umbral = cv2.threshold(restados, 20, 255, cv2.THRESH_BINARY)[1]

            heatmaps.append(umbral)
            i += 1
    j = 0
    for htmp in heatmaps:
        cv2.imshow('Frame' + str(j),cv2.flip(htmp,1))
        if not result:
            break
        j += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    if 0xFF == ord('r'):
        Capture()
    lastImgs = images
 


cam.release()
cv2.destroyAllWindows()