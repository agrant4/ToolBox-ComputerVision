import cv2
import numpy as np

cap = cv2.VideoCapture('project.avi')

while(cap.isOpened()):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    kernel = np.ones((21,21),'uint8')
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
        cv2.circle(frame,(x+w/4,y+h*4/10),(w/16),(0,0,0),-1)
        cv2.circle(frame,(x+w*3/4,y+h*4/10),(w/16),(0,0,0),-1)
        cv2.line(frame,(x+w/4,y+h/5),(x+w/2,y+h*4/10),(0,0,0),3)
        cv2.line(frame,(x+w*3/4,y+h/5),(x+w/2,y+h*4/10),(0,0,0),3)
        cv2.ellipse(frame,(x + w/2, int(y + .75*h)),(50,25),0,0,180,(0, 0, 0), thickness=4)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
