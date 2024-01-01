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

cam = cv2.VideoCapture(0)

People = []
PeopleAmountInit = 0

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
   result, image = cam.read()
   h, w, c = image.shape
   results = model(image, size=1980)
   Predictions = results.pred[0] # 0 because it is image 1
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
   print(People[0])


Capture()
while True:

    result, image = cam.read()
    results = model(image)

    Final = np.squeeze(results.render())
    if result == True:
       
       cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE) 
       cv2.imshow('Frame',Final)

        # Press Q on keyboard to  exit
       if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    else: 
        break
 


cam.release()
cv2.destroyAllWindows()