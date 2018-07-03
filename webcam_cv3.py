import cv2
import sys
import logging as log
import datetime as dt
import time

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

angle_check = 30 #angle in which to check for rotated faces

period = 0.1 # loop at 10Hz
t = time.time()

while True:
    t+=period
    
    if not video_capture.isOpened():
        print('Unable to load camera.')
        time.sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))    #try to detect face
    
    if (len(faces) == 0):                                                 #if no faces, try rotating 20 degrees
        rows,cols = gray.shape
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),angle_check,1)
        gray = cv2.warpAffine(gray,M,(cols,rows))
        
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
        
    if (len(faces) == 0):                                                 #if still nothing, try rotating the other way
        rows,cols = gray.shape
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),(360 - 2*angle_check),1)
        gray = cv2.warpAffine(gray,M,(cols,rows))
        
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', gray)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', gray)
    
    #sleep until time passes
    time.sleep(max(0,t-time.time()))

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
