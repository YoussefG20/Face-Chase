import cv2 
import time
import random 

cap = cv2.VideoCapture(0)
last_switch_time = time.time()
switch_interval = 5


 
face_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_alt.xml')

def detect_face (img):
    plate_img = img.copy()
    plate_rects = face_cascade.detectMultiScale(plate_img,scaleFactor=1.3, minNeighbors=3)
    roi = img.copy()

    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img,(x,y), (x+w,y+h), (0,0,255), thickness=4)    

    return plate_img, plate_rects

cv2.namedWindow('Test')


position = (0,0)
score = 0
while True: 
    ret, frame = cap.read()

    height = frame.shape[0]
    width = frame.shape[1]

    
    if time.time() - last_switch_time >= switch_interval:
        position = (random.randint(0, width - 1), random.randint(0, height - 1)) # Example alternate position
        last_switch_time = time.time()  # Update last switch time

    cv2.putText(frame, "SCORE:" + str(score), (0,60), cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(255,0,0), thickness=2)
    cv2.circle(frame, center=position, radius=6, color=(0, 0, 255), thickness=-1)

    detected_frame, plate_rects = detect_face(frame)
    
    for (x, y, w, h) in plate_rects:
        if x < position[0] < x + w and y < position[1] < y + h:
            cv2.putText(detected_frame, 'Circle inside face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            position = (random.randint(0, width - 1), random.randint(0, height - 1)) 
            score = score + 1 
    
    cv2.imshow('Test', detected_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


cap.release()
cv2.destroyAllWindows()