#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Original : Adrian Rosebrock
Modified : Nicolas Eberlé

"""

# import the necessary packages
from threading import Thread
import sys
import cv2
import time
import numpy as np

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue

# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue




class FileVideoStream:
    def __init__(self, path, flag_transform=None, queueSize=64,roi=None):
        self.stream = cv2.VideoCapture(path)
        self.frame_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stopped = False
        self.flag_transform = flag_transform
        #initialize the camera mapping
        if self.flag_transform == 1:
            self.mtx = np.load('mtx_file.npy')
            self.dist = np.load('dist_file.npy')
            self.newcameramtx = np.load('newcameramtx_file.npy')
            self.mapx,self.mapy = cv2.initUndistortRectifyMap(self.mtx,self.dist,None,self.newcameramtx,(self.frame_width,self.frame_height),5)
			# we load the pre_roi due to undistortion of the image
            self.roi = np.load('roi_file.npy')
            self.x_roi,self.y_roi,self.w_roi,self.h_roi = self.roi
			#
			#if a roi is defined, we have to consider the previous roi
        if roi != None:
                x1 = roi[1]
                x2 = roi[2]
                self.x_roi += x1[0]#due to previsou roi
                self.y_roi += x1[1]#due to previsou roi
                self.w_roi = abs(x2[0]-x1[0])
                self.h_roi = abs(x2[1]-x1[1])

		# initialize the queue used to store frames read from
		# the video file
        self.Q = Queue(maxsize=queueSize)
    def start(self):
		# start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    def update(self):
		# keep looping infinitely
        while True:
			# if the thread indicator variable is set, stop the
			# thread
            if self.stopped:
                return

			# otherwise, ensure the queue has room in it
            if not self.Q.full():
				# read the next frame from the file
                (grabbed, frame) = self.stream.read()
                frame_pos_msec = self.stream.get(cv2.CAP_PROP_POS_MSEC)
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
                if not grabbed:
                    self.stop()
                    return

				# if there are transforms to be done, might as well
				# do them on producer thread before handing back to
				# consumer thread. ie. Usually the producer is so far
				# ahead of consumer that we have time to spare.
				#
				# Python is not parallel but the transform operations
				# are usually OpenCV native so release the GIL.
				#
				# Really just trying to avoid spinning up additional
				# native threads and overheads of additional
				# producer/consumer queues since this one was generally
				# idle grabbing frames.
                if self.flag_transform:
					#frame = self.transform(frame)
                    frame_f1 = cv2.remap(frame,self.mapx,self.mapy,cv2.INTER_LINEAR)
                    frame = frame_f1[self.y_roi:self.y_roi+self.h_roi, self.x_roi:self.x_roi+self.w_roi]

				# add the frame to the queue
                self.Q.put([frame_pos_msec,frame])
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

    def read(self):
		# return next frame in the queue
        return self.Q.get()

	# Insufficient to have consumer use while(more()) which does
	# not take into account if the producer has reached end of
	# file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
		# return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
		# indicate that the thread should be stopped
        self.stopped = True
