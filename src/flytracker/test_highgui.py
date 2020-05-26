import cv2

def nothing(x):
    pass

cap = cv2.VideoCapture('./dataset/out.mp4')

if (cap.isOpened()== False): 
    print("Error opening video stream or file")
    exit(0)
cv2.namedWindow('frame')
cv2.createTrackbar('Seuil','frame',0,255,nothing)

while True:
    ret,frame_full = cap.read()
    frame = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame",frame)
    cv2.waitKey(1)


