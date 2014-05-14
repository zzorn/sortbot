
import numpy as np
import cv2
import cv2.cv as cv

def nothing(x):
    pass


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')




cap = cv2.VideoCapture(0)


# create trackbars
uiImage = np.zeros((200,512,3), np.uint8)
cv2.namedWindow('uiFrame')
cv2.createTrackbar('P1','uiFrame',0,300,nothing)
cv2.createTrackbar('P2','uiFrame',0,300,nothing)
cv2.createTrackbar('Min Distance','uiFrame',10,800,nothing)
cv2.createTrackbar('Min Radius','uiFrame',10,800,nothing)
cv2.createTrackbar('Max Radius','uiFrame',10,800,nothing)
cv2.createTrackbar('Blurring','uiFrame',0,20,nothing)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #cv2.rectangle(colorImg,(10,10),(20,20),(255,0,255),2)



    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    circleP1 = max(1, cv2.getTrackbarPos('P1','uiFrame'))
    circleP2 = max(1, cv2.getTrackbarPos('P2','uiFrame'))
    circleDist = max(1, cv2.getTrackbarPos('Min Distance','uiFrame'))
    minRadius = max(1, cv2.getTrackbarPos('Min Radius','uiFrame'))
    maxRadius = max(1, cv2.getTrackbarPos('Max Radius','uiFrame'))
    blurring = max(1, cv2.getTrackbarPos('Blurring','uiFrame'))


#    gray = cv2.medianBlur(gray,5)
#    gray = cv2.medianBlur(gray,5)
#    gray = cv2.medianBlur(gray,5)
#    for i in range(blurring):         
#        gray = cv2.medianBlur(gray,5)
        


    # Find faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Find circles
#    circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT , 4, circleDist, param1=circleP1,param2=circleP1,minRadius=minRadius,maxRadius=maxRadius)

    colorImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

    #circles = np.uint16(np.around(circles))
#    if (circles != None): 
 #       for i in circles[0,:]:
  #          # draw the outer circle
   #         cv2.circle(colorImg,(i[0],i[1]),i[2],(0,255,0),2)
    #        # draw the center of the circle
     #       cv2.circle(colorImg,(i[0],i[1]),2,(0,0,255),3)

    #print( "Faces", len(faces) )
    for (x,y,w,h) in faces:
        cv2.rectangle(colorImg,(x,y),(x+w,y+h),(255,0,0),2)
                
        # Create region of interest of face, find eyes in it       
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = colorImg[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    # Display the resulting frame
    cv2.imshow('frame',colorImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('uiFrame',uiImage)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


