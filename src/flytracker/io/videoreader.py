# import the necessary packages
from threading import Thread
from cv2 import VideoCapture
from queue import Queue


class VideoReader:
    def __init__(self, video_path, max_queue=50):
        self.stream = VideoCapture(video_path)
        self.queue = Queue(maxsize=max_queue)
        self.stop_queue_updating = False  # stops loading images

        # Starting right away in separate thread
        Thread(target=self._update_queue, daemon=True).start()

    def _update_queue(self):
        # keep looping infinitely
        while True:
            if self.stop_queue_updating:
                break
            if not self.queue.full():
                succes, frame = self.stream.read()
                if succes is False:
                    self.stop()
                self.queue.put((succes, frame))

    def read(self):
        # If the queue is empty and the stream is finished and you try to read
        # it'll wait infinitely long, so we return (False, None) like opencv.
        if self.queue.empty() and self.stop_queue_updating:
            return False, None
        else:
            return self.queue.get()

    def stop(self):
        self.stop_queue_updating = True

