from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
import math

def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

#Color boundaries

#green    
color1lower = (29, 86, 6)
color1upper = (64, 255, 255)

#red
color2lower = (76, 0, 153)
color2upper = (204, 204,255)

angleTolerance = 5
 
#Reference to webcam
vs = VideoStream(src=0).start()
#time.sleep(1.0)

#Keep looping
while True:
    
        #Frame grab
        frame = vs.read()
        frame
        if frame is None:
            break
  
        #Image preparation
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
 
        #Construct masks and smooth image
        mask1 = cv2.inRange(hsv, color1lower, color1upper)
        mask1 = cv2.erode(mask1, None, iterations=2)
        mask1 = cv2.dilate(mask1, None, iterations=2)

        mask2 = cv2.inRange(hsv, color2lower, color2upper)
        mask2 = cv2.erode(mask2, None, iterations=2)
        mask2 = cv2.dilate(mask2, None, iterations=2)
               
        #Find contours and initialize variables
        cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = cnts1[0] if imutils.is_cv2() else cnts1[1]

        cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = cnts2[0] if imutils.is_cv2() else cnts2[1]
        
        center1a = None
        center1b = None
        center2a = None
        center2b = None
        angle = 0


        #If a contour is found (first color)
        if len(cnts1) > 0:
                
                #Find largest contour & establish circle
                c1a = max(cnts1, key=cv2.contourArea)              
                ((x1a, y1a), radius1a) = cv2.minEnclosingCircle(c1a)
                M1a = cv2.moments(c1a)
                center1a = (int(M1a["m10"] / M1a["m00"]), int(M1a["m01"] / M1a["m00"]))
               
                #If radius is big enough, draw circle
                if radius1a > 1:
                        cv2.circle(frame, (int(x1a), int(y1a)), int(radius1a),
                        (0, 255, 255), 2)
                        cv2.circle(frame, center1a, 5, (0, 0, 255), -1)
                #Find second point
                if len(cnts1) > 1:
                        removearray(cnts1,c1a)
                        c1b = max(cnts1, key=cv2.contourArea)              
                        ((x1b, y1b), radius1b) = cv2.minEnclosingCircle(c1b)
                        M1b = cv2.moments(c1b)
                        center1b = (int(M1b["m10"] / M1b["m00"]), int(M1b["m01"] / M1b["m00"]))

                        if radius1b > 1:
                                cv2.circle(frame, (int(x1b), int(y1b)), int(radius1b),
                                (0, 255, 255), 2)
                                cv2.circle(frame, center1b, 5, (0, 0, 255), -1)
                                
        #If a contour is found (second color)
        if len(cnts2) > 0:
                
                #Find largest contour & establish circle
                c2a = max(cnts2, key=cv2.contourArea)              
                ((x2a, y2a), radius2a) = cv2.minEnclosingCircle(c2a)
                M2a = cv2.moments(c2a)
                center2a = (int(M2a["m10"] / M2a["m00"]), int(M2a["m01"] / M2a["m00"]))
               
                #If radius is big enough, draw circle
                if radius2a > 1:
                        cv2.circle(frame, (int(x2a), int(y2a)), int(radius2a),
                        (0, 255, 255), 2)
                        cv2.circle(frame, center2a, 5, (0, 0, 255), -1)
                #Find second point
                if len(cnts2) > 1:
                        removearray(cnts2,c2a)
                        c2b = max(cnts2, key=cv2.contourArea)              
                        ((x2b, y2b), radius2b) = cv2.minEnclosingCircle(c2b)
                        M2b = cv2.moments(c2b)
                        center2b = (int(M2b["m10"] / M2b["m00"]), int(M2b["m01"] / M2b["m00"]))

                        if radius2b > 1:
                                cv2.circle(frame, (int(x2b), int(y2b)), int(radius2b),
                                (0, 255, 255), 2)
                                cv2.circle(frame, center2b, 5, (0, 0, 255), -1)                       

        #Show tracked positions on screen
        cv2.putText(frame, "Center 1a: {}".format(center1a),
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)
        
        cv2.putText(frame, "Center 1b {}".format(center1b),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

        cv2.putText(frame, "Center 2a: {}".format(center2a),
                (140, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)
        
        cv2.putText(frame, "Center 2b {}".format(center2b),
                (140, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

        #Checking if a second point exists
        if len(cnts1) > 1:
            cv2.line(frame, center1a, center1b, (0, 0, 255), 2)
            deltay1 = y1b-y1a
            deltax1 = x1b-x1b
            if (deltax1 == 0):
                m1 = 0
            else:    
                m1 = (deltay1)/(deltax1)
            b1 = (y1a-m1*x1a)
            

        if len(cnts2) > 1:
            cv2.line(frame, center2a, center2b, (0, 0, 255), 2)
            deltay2 = y2b-y2a
            deltax2 = x2b-x2b
            if (deltax2 == 0):
                m2 = 0
            else:    
                m2 = (deltay2)/(deltax2)
            b2 = (y2a-m2*x2a)

        #If both second points exist, find elbow and calculate angle
        if (len(cnts1) > 1 and len(cnts2) > 1):
            if (m1 != m2):
                xe = (b2-b1) / (m1-m2)
                ye = m1*xe+b1
                e = (xe, ye)
                cv2.line(frame, center1a, e, (0, 0, 255), 2)
                cv2.line(frame, center1b, e, (0, 0, 255), 2)
                cv2.line(frame, center2a, e, (0, 0, 255), 2)
                cv2.line(frame, center2b, e, (0, 0, 255), 2)

                angle = math.degrees(math.pi - math.fabs(math.atan(m1) - math.atan(m2)))

            #Indicate whether 90˚ has been reached
                if ((math.fabs(angle-90)) < angleTolerance):
                    angleColor = (255, 0, 0)
                else:
                    angleColor = (0, 0, 255)
            
            
                cv2.putText(frame, "{}˚".format(angle),
                    (270, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 5) #make 5 1
            
        cv2.imshow("Frame", frame)           
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord("q"):
                break
 
vs.stop()
cv2.destroyAllWindows()
