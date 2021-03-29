import pandas as pd
from flytracker.analysis.annotating import annotate

# Should always be mp4 - h264 doesnt set frame correctly.
movie_path = (
    "/home/gert-jan/Documents/flyTracker/data/experiments/bruno/videos/seq_1.mp4"
)
data_path = "data/experiments/bruno/results/df.hdf"
mapping_folder = "data/distortion_maps/"
output_loc = "data/experiments/bruno/videos/annotated_video.mp4"
df = pd.read_hdf(data_path, key="df")

annotate(
    df,
    movie_path,
    mapping_folder,
    output_loc,
    max_frames=None,
    track_length=20,
    touching_distance=10,
)
