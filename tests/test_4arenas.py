from flytracker.tracker import Tracker
from flytracker.utils import FourArenasQRCodeMask

mask = FourArenasQRCodeMask()
tracker = Tracker(40, mask)
tracker.run('data/movies/4arenas_QR.h264', 1000)