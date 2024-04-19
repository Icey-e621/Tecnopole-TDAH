import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sympy import frac, true
import yolov5
import torch
import zipfile
from PIL import Image, ImageTk
import tkinter as tk
import time


# load pretrained model
model = yolov5.load("yolov5m.pt")
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
model.cpu  # i.e. device=torch.device(0)

model2 = tf.keras.models.load_model('TDAH.keras')

cam = cv2.VideoCapture(0)

People = []
PeopleAmountInit = 0
lastImgs = []
Ticks = 0

################################################################
##################    Classes   ################################
################################################################

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
   
Output = 0
def Recognize(images):

    Data = [cv2.resize(i,(250,250),interpolation=Image.LANCZOS) for i in images]



TimesCalled = 0
def Capture():
#   lastImgs = []
#   Ticks = 0 
   global TimesCalled
   TimesCalled = 1 + TimesCalled
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
       if TimesCalled == 1:
           People.append(Person(f'Joe {str(Count)}',box))
       else:
            try:
                if not People or Person("Joe " + str(Count),box) == People[Count-1]:
                   People.append(Person("Joe " + str(Count),box))

                elif Person("Joe " + str(Count),box) != People[Count-1]:
                   print("t")
                   People.insert(Count,Person("Joe " + str(Count),box) == People[Count-1])

            except os.error:
                 print(os.error)

def timer(timerdisplay, t,image):
    Recon = true
    root.itemconfigure(timerdisplay, text=t)
    if t >= 1:
        root.after(1000, timer, timerdisplay, t-1)

Recon = False
Capture()
while True:
    Ticks += 1
    result, image = cam.read()
    images = []
    global heatmaps; heatmaps = []
    tmp = []
    fig, ax = plt.subplots()

    for Pers in People:
        global STOP; STOP = False
        
        try:
            images.append(image[int(Pers.box.y1):int(Pers.box.y2),int(Pers.box.x1):int(Pers.box.x2)])
        except:
            STOP = true
            break


    i = 0
    if Ticks > 1:
        for img in images:
            grisNow = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            grisBefore = cv2.cvtColor(lastImgs[i], cv2.COLOR_BGR2GRAY)

            restados = cv2.absdiff(grisBefore, grisNow)

            umbral = cv2.threshold(restados, 25, 255, cv2.THRESH_BINARY)[1]
            umbral = cv2.dilate(umbral, None, iterations=2)

            contours, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # Get the bounding box of the contour
                x, y, w, h = cv2.boundingRect(umbral)
                
                # If the contour is too small, skip it
                if w > img.shape[1]-30 or h > img.shape[0]-30:
                    cv2.rectangle(umbral,(x,y),(x+w,y+h),(0,255,0),-1)
                    continue



            heatmaps.append(umbral)
            i += 1
        h = 0
    


        # Create the main window
        root = tk.Tk()

        big_image = image
        for Pers in People:
            cv2.rectangle(big_image,(int(Pers.box.y1),int(Pers.box.y2)),(int(Pers.box.x1),int(Pers.box.x2)),(0,255,0),1)

        big_image_width = root.winfo_screenwidth() // 2
        big_image_height = root.winfo_screenheight() // 2
        big_image = cv2.resize(big_image,(big_image_width, big_image_height),interpolation=Image.LANCZOS)

        # Create a label to display the big image
        big_image_label = tk.Label(root, image=ImageTk.PhotoImage(Image.fromarray(big_image)),text="Person Recognition")
        big_image_label.pack(pady=50)
        


        # Load the smaller images
        smaller_images = heatmaps

        smaller_image_width = 150
        smaller_image_height = int(smaller_image_width // (big_image_width / big_image_height))
        smaller_images = [cv2.resize(image,(smaller_image_width, smaller_image_height),interpolation=Image.LANCZOS) for image in smaller_images]

        
        # Create a frame to hold the smaller images
        smaller_image_frame = tk.Frame(root)
        smaller_image_frame.pack(pady=50)

        

        # Create a grid to hold the smaller images

        tk.Label(root, text="Personas en la pantalla en mapas de movimiento (mantenganse quietos)").pack(pady=20)

        # Display the smaller images in the grid
        for i, image in enumerate(smaller_images):
            tk_image = ImageTk.PhotoImage(Image.fromarray(image))
            tk.Label(root, image=tk_image).grid(row=i // 3, column=i % 3)

            

        # Run the main loop
        root.mainloop()

        #for htmp in heatmaps:
        #    h += 1
        #    cv2.imshow
            

        
        #if cv2.waitKey(0) & 0xFF == ord('q'):
        #    break

    lastImgs = images
 

plt.close()
cam.release()
cv2.destroyAllWindows()