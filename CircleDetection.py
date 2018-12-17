import sys
import cv2 as cv
import numpy as np
from filevideostream import FileVideoStream

def main(argv):
    
    default_file = 'thumb.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(filename, cv.IMREAD_COLOR)
    height,width,channel = src.shape
    scale = 2.5
    src = cv.resize(src,(int(width/scale), int(height/scale)), interpolation = cv.INTER_LINEAR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=20, param2=50,
                               minRadius=int(80/scale), maxRadius=int(140/scale))
    if circles.shape[1] == 4:
        index_roi = 1
        for i in circles[0, :]:
            x1 = (int(i[0]-200/scale), int(i[1]-200/scale))
            x2 = (int(i[0]+200/scale), int(i[1]+200/scale))
            cv.rectangle(src,x1,x2,(0,255,0),2)
            str_roi = str(index_roi)
            index_roi = index_roi +1
            cv.imshow(str_roi, src[x1[1]:x2[1],x1[0]:x2[0]])            
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            print(center)
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
    
    
    cv.imshow("detected circles", src)
    cv.waitKey(0)
    
if __name__ == "__main__":
    main(sys.argv[1:])
