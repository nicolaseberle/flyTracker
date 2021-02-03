# %%
import cv2
import numpy as np
from os.path import join
from seaborn import color_palette
from scipy.spatial import distance_matrix


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
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        filename=output_loc, fourcc=fourcc, fps=fps, frameSize=image_size, isColor=color
    )
    return writer


def update_mask(mask, df_n_minus_one, df_n):
    def write_mask(mask, flies):
        # flies should be list with entry for each fly,
        # each entry should be a tuple of tuples.
        palette = color_palette("Paired")
        color = lambda idx: tuple(color * 255 for color in palette[idx % len(palette)])

        for idx, position in enumerate(flies):
            mask = cv2.line(mask, position[0], position[1], color(idx), thickness=1,)
        return mask

    # Turn dataframe data into tuple ready to give to opencv
    df_to_tuple = lambda df: tuple(df[["x", "y"]].to_numpy(dtype=np.int32).squeeze())
    # Turn data into list of tuples of tuples
    # List corresponds to flies, outer tuple to old/new position
    # inner tuple to (x, y)

    fly_locs = [
        (df_to_tuple(df_n_minus_one_ID), df_to_tuple(df_n_ID))
        for (_, df_n_minus_one_ID), (_, df_n_ID) in zip(
            df_n_minus_one.groupby("ID"), df_n.groupby("ID")
        )
    ]

    mask = write_mask(mask, fly_locs)
    return mask


def write_ID(image, df_n, touching_distance=15):
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

    # Get (ID, location) tuple
    df_to_tuple = lambda df: tuple(df[["x", "y"]].to_numpy(dtype=np.int32).squeeze())
    fly_ID = [(int(ID), df_to_tuple(df_n_ID)) for ID, df_n_ID in df_n.groupby("ID")]

    # Find out if they're touching
    local_locs = df_n[["x", "y"]].to_numpy()
    dist_matrix = distance_matrix(local_locs, local_locs)
    # minimum 2 cause diagonal element are always 0
    touching = np.sum(dist_matrix < touching_distance, axis=0) >= 2

    for (ID, loc) in fly_ID:
        image = add_fly_ID(image, loc, ID, touching[ID])
    return image

