import numpy as np
import cv2
import glob

img = cv2.imread('./chessboard/image_test_calibration.jpg')
h,  w = img.shape[:2]

mtx = np.load('mtx_file.npy')
dist = np.load('dist_file.npy')
newcameramtx = np.load('newcameramtx_file.npy')
roi = np.load('roi_file.npy')

mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)

dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# crop the image 
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('RemapCalibResult.png',dst)

cv2.imshow('out',dst)
cv2.waitKey(0)

