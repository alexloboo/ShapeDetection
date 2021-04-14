import cv2
import numpy as np

cara_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ojo_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
sonrisa_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    caras = cara_cascade.detectMultiScale(gris, 1.3 ,10)
    for(x,y,w,h) in caras:
        cv2.rectangle(img, (x,y), (x+w, y+h),(255, 255, 0),2)
        roi_gris = gris[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        ojos = ojo_cascade.detectMultiScale(roi_gris)
        for (ex,ey,ew,eh) in ojos:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0, 255, 255), 2)
    
    sonrisas  = sonrisa_cascade.detectMultiScale(img, scaleFactor = 1.8, minNeighbors = 20)
    for (x, y, w, h) in sonrisas:
            cv2.rectangle(img, (x, y), ((x + w), (y + h)), (0, 255,0), 5)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()