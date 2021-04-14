# %%
import cv2
import numpy as np
import os

from itertools import takewhile
from scipy.spatial import distance_matrix
from seaborn import color_palette

from ..io import DataLoader


def annotate(
    df, movie_path, output_loc, max_frames=None, track_length=30, touching_distance=10,
):

    # Parsing dataframe to numpy array - much faster
    data, n_flies_per_arena = parse_data(df)
    track_length = int(np.around(track_length * 30))
    color_fn = lambda ID: color_picker(ID, n_flies_per_arena)

    # Making dataset
    initial_frame = data[0, 0, 0]
    n_frames = data.shape[0] if max_frames is None else max_frames

    assert movie_path.split(".")[-1] == "mp4", "Movie should be mp4."
    loader = DataLoader(movie_path, parallel=False)
    # plus 1 for intiial frame since we plot (n-1, n)
    loader.dataset.set_frame(initial_frame + 1)

    # Setting up loader and writer
    writer = None

    for idx, image in takewhile(lambda x: x[0] < n_frames, enumerate(loader, start=1)):
        image = image.numpy().squeeze()
        lower_frame, upper_frame = np.maximum(idx - track_length, 0), idx

        image = add_frame_info(image, f"frame: {upper_frame}")
        # First write tracks so that numbers don't get occluded.
        image = write_tracks(image, data[lower_frame:upper_frame], color_fn)
        image = write_ID(image, data[upper_frame], touching_distance=touching_distance)
        if writer is None:
            # first two are image size - somehow we need to invert shapes to get opencv to write
            writer = setup_writer(output_loc, image.shape[:2][::-1], fps=30)

        writer.write(image)

        if idx % 1000 == 0:
            print(f"Done with frame {idx}")
    writer.release()

    # Compressing to h264 with ffmpeg
    compressed_loc = output_loc.split(".")[0] + "_compressed.mp4"
    os.system(f"ffmpeg -i {output_loc} -an -vcodec libx264 -crf 23 {compressed_loc}")


def parse_data(df):
    # Changing to numpy array for speed
    df = df.sort_values(by=["frame", "arena", "ID"])
    n_flies = df.ID.unique().size
    n_features = df.shape[1]
    data = df.to_numpy().reshape(-1, n_flies, n_features)
    data = np.around(data).astype(int)  # everything must happen with ints

    max_n_flies_per_arena = np.max(
        [df.query(f"arena == {arena}").ID.unique().size for arena in df.arena.unique()]
    )

    return data, max_n_flies_per_arena


def setup_writer(output_loc, image_size, fps=30, color=True):
    """Returns writer object."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        filename=output_loc, fourcc=fourcc, fps=fps, frameSize=image_size, isColor=color
    )
    return writer


def add_frame_info(img, text):
    """ Add frame number to upper left corner."""
    return cv2.putText(
        img,
        text,
        org=(50, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 0),
        thickness=2,
    )


def write_ID(image, frame_info, touching_distance=10):
    def add_fly_ID(image, loc, ID, touching):
        if touching == True:
            color = (0, 0, 255)  # red
        else:
            color = (255, 0, 0)  # blue

        return cv2.putText(
            image,
            text=f"{ID}",
            org=(loc[0] - 5, loc[1] - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1,
        )

    dist_matrix = distance_matrix(frame_info[:, [2, 3]], frame_info[:, [2, 3]])
    # minimum 2 cause diagonal element are always 0
    touching = np.sum(dist_matrix < touching_distance, axis=0) >= 2
    # we just iterate over the rows
    for fly in frame_info:
        image = add_fly_ID(image, fly[[3, 2]], fly[1], touching[fly[1]])
    return image


def write_tracks(image, data, color_fn):
    for idx in np.arange(data.shape[1]):
        image = cv2.polylines(image, [data[:, idx, [3, 2]]], False, color_fn(idx), 1)
    return image


def color_picker(ID, n_flies_per_arena, palette=color_palette("Paired")):
    assert n_flies_per_arena < len(
        palette
    ), "More flies per arena than colors in palette."
    # turning into rgb for opencv
    color = tuple(color * 255 for color in palette[ID % n_flies_per_arena])
    return color

