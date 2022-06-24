# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 02:56:41 2019

@author: Elton
"""
import cv2
import random
import numpy as np
from keras.models import load_model
import urllib
import time
modelN = load_model('facial_1')
EMOTION_DICT = {1:"ANGRY", 2:"FEAR", 3:"HAPPY", 4:"SAD", 5:"SURPRISE", 6:"NEUTRAL"}
from scipy import stats
from keras_preprocessing import image
from keras.preprocessing.image import img_to_array, array_to_img
import pickle
time.sleep(0.1)

modelNl = load_model('VGG_cross_validated.h5')
PLAY_DICT = {0:"STONE", 1:"PAPER", 2:"SCISSOR"}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.read("trainner.yml")
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}
i=0
flag=0
scan=0
import socket
#client = socket.socket()	
#port = 12345
#client.connect(('192.168.1.106', port))
#
#time.sleep(1)
#client.settimeout(0.001)




import os
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

PLAY_DICT = {0: 'Stone',
                 1: 'Scissor',
                 2: 'Paper',
                 3: 'Paper',
                 4: 'Scissor'}
				 
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
hand_text="None"

isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works

flag=0
cap = cv2.VideoCapture("http://192.168.1.107/html/cam_pic_new.php?time=1572087388503&pDelay=40000")

fgbg = cv2.createBackgroundSubtractorMOG2()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29,86,6)
greenUpper = (64,255,255)
#greenLower = (3, 50, 50)
#greenUpper = (33, 255, 255)
#lower = np.array([0, 48, 80], dtype = "uint8")
#upper = np.array([20, 255, 255], dtype = "uint8")
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	#vs = VideoStream(src=0).start()
	print("not using pc webcam")

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up






s = socket.socket()

#		 
print ("Socket successfully created")

# reserve a port on your computer in our 
# case it is 12345 but it can be anything 
port = 12345				

# Next bind to the port 
# we have not typed any ip in the ip field 
# instead we have inputted an empty string 
# this makes the server listen to requests 
# coming from other computers on the network 	 
s.bind(('', port))
print ("socket binded to %s" %(port)) 

# put the socket into listening mode 
s.listen(5)	 
print ("socket is listening")	

client, addr = s.accept()	 
print ('Got connection from', addr) 
#        c.send(str.encode(str(1)))
client.settimeout(0.001)

#print(1)










def return_prediction(path):
    #converting image to gray scale and save it
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, gray)
    
    #detect face in image, crop it then resize it then save it
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face_clip = img[y:y+h, x:x+w]
        cv2.imwrite(path, cv2.resize(face_clip, (350, 350)))
    
    #read the processed image then make prediction and display the result
    #detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
    img1 = cv2.imread(path)
    detected_face = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #transform to gray scale
    detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
    img_pixels = image.img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255
    print("Reached Here")
    
    y_prob = modelN.predict(img_pixels[0:1])
    y_pred = [np.argmax(prob) for prob in y_prob]
    #y_true = [np.argmax(true) for true in y]
    emotion_label = y_pred[0].argmax() + 1
    
    
    print(y_prob,y_pred[0])
    print(emotion_label)
    return (y_pred[0])



     
def quad_run(text,cap):
    global i, flag, scan, face_cascade, recognizer, labels
    arr=[]
    lf=time.time()
    m=0
    name= ""
    mode=1
    while True:
        ret, img = cap.read()
        hh,ww,c = img.shape
        h1=int(hh/3)
        h2=int(2*hh/3)
        w1=int(ww/3)
        w2=int(2*ww/3)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX

#        cv2.line(img,(0,h1),(ww,h1),(0,100,50),2)
#        cv2.line(img,(0,h2),(ww,h2),(0,100,50),2)
#        cv2.line(img,(w1,0),(w1,hh),(0,100,50),2)
#        cv2.line(img,(w2,0),(w2,hh),(0,100,50),2)

        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        now=int(round(time.time() * 1000))
        if len(faces)==0:
            if (now-lf>4000 and scan==0):
#                print("First serial write")
#                if scan==0:
#                    ser.write(b'0')
                client.send(str.encode(str(0)))
#                ser.write(b'2')
#                client.send(str.encode(str(2)))
                scan=1
                
        for x,y,w,h in faces:
            lf=now
            scan=0
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
            midx=int((x+x+w)/2)
            midy=int((y+y+h)/2)
            
            
        if scan==0:
#            ser.write(b'0')
#            client.send(str.encode(str(0)))
#            ser.write(b'1')
#            client.send(str.encode(str(1)))
            #find placement of mid in grid
            if(midx in range(0,w1)):
                if(midy in range(0,h1)):
                    #Q1
 #                   ser.write(b'1')t
                    client.send(str.encode(str(1)))
                    cv2.putText(img, "Last Quandrant was 1", (95,30), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                elif(midy in range(h1,h2)):
                    #Q4
  #                  ser.write(b'4')
  
                    cv2.putText(img, "Last Quandrant was 4", (95,30), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                    client.send(str.encode(str(4)))
                else:
                    #Q7
   #                 ser.write(b'7')
                    cv2.putText(img, "Last Quandrant was 7", (95,30), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                    client.send(str.encode(str(7)))
            elif(midx in range(w1,w2)):
                if(midy in range(0,h1)):
                    #Q2
    #                ser.write(b'2')
                    cv2.putText(img, "Last Quandrant was 2", (95,30), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                    client.send(str.encode(str(2)))
                elif(midy in range(h1,h2)):
                    #Q5
     #               ser.write(b'5')
                    cv2.putText(img, "Last Quandrant was 5", (95,30), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                    client.send(str.encode(str(5)))
                    if flag==1 and time.time()-stime>0.5:
                        stime=time.time()
                        cv2.imwrite('photo'+str(i)+'.jpg',img)
                        i+=1
                    if mode==3 and time.time()-rtime>0.5:
                        rtime=time.time()
                        for (x,y,w,h) in faces:
                            face_clip = gray[y:y+h, x:x+w]
                            cv2.imwrite(os.path.join(paths,'photo'+str(m)+'.jpg'),cv2.resize(face_clip, (350, 350)))
                        m+=1
                    
                else:
                    #Q8
      #              ser.write(b'8')
                    cv2.putText(img, "Last Quandrant was 8", (95,30), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                    client.send(str.encode(str(8)))
            else:
                if(midy in range(0,h1)):
                    #Q3
       #             ser.write(b'3')
                    cv2.putText(img, "Last Quandrant was 3", (95,30), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                    client.send(str.encode(str(3)))
                elif(midy in range(h1,h2)):
                    #Q6
        #            ser.write(b'6')
                    cv2.putText(img, "Last Quandrant was 6", (95,30), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                    client.send(str.encode(str(6)))
                else:
                    #Q9
         #           ser.write(b'9')
                    cv2.putText(img, "Last Quandrant was 9", (95,30), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                    client.send(str.encode(str(9)))
                
            roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
            id_, conf = recognizer.predict(roi_gray)
            print(labels[id_],conf)
            if conf >= 44 and conf <= 110:
                   font = cv2.FONT_HERSHEY_SIMPLEX
                   name=labels[id_]
                   print()
                
            else: 
                name = "Unknown"
        cv2.putText(img, "Hold Q: To Quit", (460,470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img,name+ "'s Last Emotion was ", (5,60), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "Press SPACE: FOR EMOTION", (5,470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img,str(text), (0,160), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

        
        cv2.imshow("Image", img)
        
        
    	# clear the stream in preparation for the next frame
        if i>20:
            i=0
            arr=np.ones(20)
            for j in range(20):
                arr[j]=(return_prediction('photo'+str(j)+'.jpg'))
            text="x"+EMOTION_DICT[int(stats.mode(arr)[0])+1]+"z"
            client.send(str.encode(text))
            flag=0
            
        if m>30:
            m=0
            mode=1
            os.system("python faces-train.py")
            time.sleep(2)
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            recognizer = cv2.face.LBPHFaceRecognizer_create()   
#recognizer = cv2.face.createLBPHFaceRecognizer()
            recognizer.read("trainner.yml")
            labels = {"person_name": 1}
            with open("labels.pickle", 'rb') as f:
                og_labels = pickle.load(f)
                labels = {v: k for k, v in og_labels.items()}
                client.send(str.encode("xRememberedz"))

            
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            time.sleep(2)
            client.close()
            time.sleep(1)
            cv2.destroyAllWindows()
            break
        
        
        try:
            mode = ord(client.recv(4096))
            print(mode)
            if mode==49:
                client.send(str.encode("t"))
                while mode==49:
                    balls()
                    try:
                        mode=ord(client.recv(4960))
                    except socket.timeout:
                        apple="a"
                        
            elif mode==51:
                if name=="Unknown":
                    print("New Face Detected, Need To Register.\n")
                    new_person = input("Enter Your Name : ")
                    paths = "D:\images\ " + new_person
                    
                    print(paths)
                    try:
                        os.makedirs(paths)
                    except OSError:
                        print("Creation Failed.")
                    else:
                        print("Creation Successful.")
                        mode=3
                        time.sleep(2)
                        rtime=time.time()
            
            elif mode==50:
                flag=1
                stime=time.time()
                text="Scanning"
                
            elif mode==52:
                name ="x"+name+"z"
                client.send(str.encode(name))
                
                
            elif mode==53:
                sps()

        except socket.timeout:
            apple="a"

        

def sps():
    import cv2
    import random
    from keras.models import load_model
    modelN = load_model('VGG_cross_validated.h5')
    PLAY_DICT = {0:"STONE", 1:"PAPER", 2:"SCISSOR"}
    import numpy as np
    from keras_preprocessing import image
    #from keras.preprocessing.image import img_to_array, array_to_img
    PLAY_DICT = {0: 'Stone',
                 1: 'Scissor',
                 2: 'Paper',
                 3: 'Paper',
                 4: 'Scissor'}
				 
    cap_region_x_begin = 0.5  # start point/total width
    cap_region_y_end = 0.8  # start point/total width
    threshold = 60  # binary threshold
    blurValue = 41  # GaussianBlur parameter
    bgSubThreshold = 50
    learningRate = 0
    hand_text="None"

    isBgCaptured = 0  # bool, whether the background captured
    triggerSwitch = False  # if true, keyboard simulator works

    flag=0

    fgbg = cv2.createBackgroundSubtractorMOG2()
    cap = cv2.VideoCapture("http://192.168.1.107/html/cam_pic.php?time=1581349819891&pDelay=40000")
    i=0
    count=2
    while(count>=0):
        cap = cv2.VideoCapture("http://192.168.1.107/html/cam_pic.php?time=1581349819891&pDelay=40000")
        
        ret, img = cap.read()
        
        if(flag==1):
            fgmask = bgModel.apply(img, learningRate=0)
        else:
                fgmask = fgbg.apply(img) 
   
    
    
    #fgmask = fgbg.apply(img)
    
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
    
        cv2.putText(img, "Last hand was "+hand_text , (95,30), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    
        cv2.rectangle(img, (int(cap_region_x_begin * img.shape[1]), 60),(img.shape[1], int(cap_region_y_end * img.shape[0])), (255, 0, 0), 2)
    #cv2.rectangle(img,int(cap_region_x_begin * img.shape[1]),int(cap_region_y_end * img.shape[0]),(0,0,255),5)
    
        cv2.putText(img, "Press SPACE: FOR PLAYING", (5,470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
        cv2.putText(img, "Hold Q: To Quit", (460,470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
    #silho = fgbg.apply(img)
    #cv2.imshow('img',img)
    
        cv2.imshow('Real', img) 
        
        cv2.imshow('withMask', fgmask)

        
        
        try:
            gdata=(client.recv(4086)).decode("utf-8")
        
        except:
            gdata="0"
        
        
        
        if gdata=="6":
            #img = fgbg.apply(img)
            #bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            #img = cv2.bilateralFilter(img, 5, 50, 100)  # smoothing filter
            #img = cv2.flip(img, 1)  # flip the frame horizontally
            #cv2.rectangle(img, (int(cap_region_x_begin * img.shape[1]), 60),(img.shape[1], int(cap_region_y_end * img.shape[0])), (255, 0, 0), 2)
            #img = remove_background(img)
            bot=random.randint(0, 2)
            client.send(str.encode(str(bot)))
            
            
        elif gdata=="7":
            img = img[60:int(cap_region_y_end * img.shape[0]), int(cap_region_x_begin * img.shape[1]):img.shape[1]]
            
            img_temp = fgmask
            img_temp=img_temp[60:350,320:720]
           
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            blur = cv2.GaussianBlur(img_temp, (blurValue, blurValue), 0)
            
            # cv2.imshow('blur', blur)
            #ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #gray=~gray
           # cv2.imwrite("test.jpg", gray)
            cv2.imwrite("test.jpg",blur)
            
            
            
            img1 = cv2.imread("test.jpg")
            #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            cv2.imshow("testing",blur)
            img1 = cv2.resize(img1, (224, 224)) #resize to 48x48
            
            img_pixels = image.img_to_array(img1)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            print(img_pixels.shape)
            y_prob = modelN.predict(img_pixels)
            y_pred = [np.argmax(prob) for prob in y_prob]
            #y_true = [np.argmax(true) for true in y]
            HAND = y_pred[0]

            print(y_prob,y_pred[0])
            print(HAND)
            print(PLAY_DICT[HAND])
            hand_text=PLAY_DICT[HAND]
            
            
            if HAND==0:
                            if bot==1:
                                client.send(str.encode(str(0)))
                            elif bot==2:
                                client.send(str.encode(str(1)))
                            elif bot==0: 
                                client.send(str.encode(str(2)))
                    
            if HAND==1:
                            if bot==2:
                                client.send(str.encode(str(0)))
                            elif bot==0:
                                client.send(str.encode(str(1)))
                            elif bot==1: 
                                client.send(str.encode(str(2)))
            
            if HAND==2:
                            if bot==0:
                                client.send(str.encode(str(0)))
                            elif bot==1:
                                client.send(str.encode(str(1)))
                            elif bot==2: 
                                client.send(str.encode(str(2)))
            print(PLAY_DICT[HAND])
            count-=1
            
            
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        elif k == ord('b'):  # press 'b' to capture the background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
            flag=1

        
def balls():
	# grab the current frame
#	vs = cv2.VideoCapture("http://192.168.1.115/elton/cam_pic.php?time=1579984175969&pDelay=40000")
    ret, frame = cap.read()
	
	#frame = vs.read()
	

	# handle the frame from VideoCapture or VideoStream
	#frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	

	# resize the frame, blur it, and convert it to the HSV
	# color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

	# only proceed if at least one contour was found
    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        c=max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
		
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
        if radius > 10:
                r=int(radius)
                print(radius)
                if(r<150 and r>120):
                    client.send(str.encode("s"))
                    dirc=0
                elif(r<120 and r>=10):
                    dirc=1
                elif(r>150):
                    client.send(str.encode("v"))
                    dirc=0
                else:
                    dirc=0
                
                if dirc==1:
                    if(center[0]<430 and center[0]>120):
                        client.send(str.encode("a"))
                    elif(center[0]<=120):
                        client.send(str.encode("l"))
                    elif(center[0]>=430):
                        client.send(str.encode("r"))
			
            
    		
			
			#image = cv2.putText(frame, radius, (50, 50) , cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA) 
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
    else:
        client.send(str.encode("s"))
	# update the points queue
    pts.appendleft(center)

	# loop over the set of tracked points
    for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
        if pts[i - 1] is None or pts[i] is None:
            continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	
	
	
	
	
	
	
	# show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

		
	# if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        return

# if we are not using a video file, stop the camera video stream


# otherwise, release the camera

# close all windows


quad_run("None", cap)