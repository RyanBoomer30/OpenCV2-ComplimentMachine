import cv2
import numpy as np
import os
import time
import playsound
import speech_recognition as sr
from gtts import gTTS
import time
import random

#Load Yolo

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
outputLayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Loading image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    height, width, channels = img.shape
    #Detecting objects
    blob = cv2.dnn.blobFromImage(img,0.00392, (416, 416), (0,0,0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(outputLayers)

    #Speech
    def speak(text):
        tts = gTTS(text=text)
        filename = "voice.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)
        time.sleep(1)

    #Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #Rectangle drawing
                x = int(center_x - w / 2)
                y = int (center_y - h / 2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    compliments = ["Your hair is very nice today", "That is a good pair of shoes u got", "You are killing it today ",
                   "I like your bag"]
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x,y+30), font, 3, color, 3)
            update = label
            if update == "person":
                x = compliments[random.randint(0,len(compliments))]
                speak(x)


    #cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
