# %%
import cv2
import numpy as np
from os.path import join

from scipy.spatial import distance_matrix
from seaborn import color_palette
import os
from itertools import count


def annotate(
    df,
    movie_loc,
    mapping_folder,
    output_loc,
    max_frames=None,
    track_length=30,
    touching_distance=10,
):

    data, n_flies_per_arena = parse_data(df)

    initial_frame = data[0, 0, 0]
    print(data.shape)
    # plus 1 for intiial frame since we plot (n-1, n)
    loader, image_size = setup_loader(
        movie_loc, mapping_folder, initial_frame=(initial_frame + 1)
    )
    writer = setup_writer(output_loc, image_size, fps=30)

    if max_frames is None:
        max_frames = 10 ** 9

    length = int(np.around(track_length * 30))

    color_fn = lambda ID: color_picker(ID, n_flies_per_arena)
    # %%
    for frame in count(start=1):
        lower_frame, upper_frame = np.maximum(frame - length, 0), frame
        image = loader()
        if (image is None) or (frame == (max_frames + 1)):
            break  # we're finished

        image = add_frame_info(image, f"frame: {upper_frame}")
        # First write tracks so that numbers don't get occluded.
        image = write_tracks(image, data[lower_frame:upper_frame], color_fn)
        image = write_ID(image, data[upper_frame], touching_distance=touching_distance)
        writer.write(image)

        if frame % 1000 == 0:
            print(f"Done with frame {frame}")
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


def construct_undistort_map(image_size, folder):
    """ Construct openCV undistort undistort mapping. Make sure files as named below are the supplied folder."""
    mtx = np.load(join(folder, "mtx_file.npy"))
    dist = np.load(join(folder, "dist_file.npy"))
    newcameramtx = np.load(join(folder, "newcameramtx_file.npy"))

    mapping = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, image_size, 5)
    return mapping


def setup_loader(movie_loc, mapping_folder, initial_frame=0):
    """Returns function which returns preprocessed image when called."""

    def load_fn(capture, mapping):
        # loads image and applies preprocessing
        ret, image = capture.read()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cause our video uses bgr
        if ret is not False:
            # image = cv2.remap(image, *mapping, cv2.INTER_LINEAR)
            pass
        else:
            image = None
            capture.release()
        return image

    # Set up capture
    cap = cv2.VideoCapture(movie_loc)
    cap.set(1, initial_frame)  # setting initial frame
    image_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # Setting up loader
    mapping = construct_undistort_map(image_size, mapping_folder)
    loader = lambda: load_fn(cap, mapping)

    return loader, image_size


def setup_writer(output_loc, image_size, fps=30, color=True):
    """Returns writer object."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        filename=output_loc, fourcc=fourcc, fps=fps, frameSize=image_size, isColor=color
    )
    return writer


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
        image = add_fly_ID(image, fly[[2, 3]], fly[1], touching[fly[1]])
    return image


def write_tracks(image, data, color_fn):
    for idx in np.arange(data.shape[1]):
        image = cv2.polylines(image, [data[:, idx, [2, 3]]], False, color_fn(idx), 1)
    return image


def color_picker(ID, n_flies_per_arena, palette=color_palette("Paired")):
    assert n_flies_per_arena < len(
        palette
    ), "More flies per arena than colors in palette."
    # turning into rgb for opencv
    color = tuple(color * 255 for color in palette[ID % n_flies_per_arena])
    return color


# %%
