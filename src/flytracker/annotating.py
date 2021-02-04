# %%
import cv2
import numpy as np
from os.path import join
from seaborn import color_palette
from scipy.spatial import distance_matrix
import pandas as pd


def parse_data(df_location):
    df = pd.read_hdf(df_location, key="df")
    df = df.sort_values(by=["frame", "ID"])
    n_flies = df.ID.unique().size
    n_features = df.shape[1]
    data = df.to_numpy().reshape(-1, n_flies, n_features)
    data = np.around(data).astype(int)  # everything must happen with ints
    return data


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
            image = cv2.remap(image, *mapping, cv2.INTER_LINEAR)
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


def update_mask(mask, frame_info_i, frame_info_j):
    palette = color_palette("Paired")
    color = lambda idx: tuple(color * 255 for color in palette[idx % len(palette)])

    for fly_frame_i, fly_frame_j in zip(frame_info_i, frame_info_j):
        mask = cv2.line(
            mask,
            tuple(fly_frame_i[[2, 3]]),
            tuple(fly_frame_j[[2, 3]]),
            color(fly_frame_j[1]),
            thickness=1,
        )
    return mask


def write_ID(image, frame_info, touching_distance=15):
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
