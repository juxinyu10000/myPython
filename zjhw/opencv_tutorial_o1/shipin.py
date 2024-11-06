import cv2

cap = cv2.VideoCapture(r'D:\Ul\MP4.mp4')

if not cap.isOpened():
    exit()
while True:
    ret,frame = cap.read()
    if not ret:
        break
    cv2.imshow('fr',frame)

cap.release()
cv2.destroyAllWindows()