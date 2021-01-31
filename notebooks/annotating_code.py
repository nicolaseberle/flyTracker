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


def setup_loader(movie_loc, mapping_folder):
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


def update_mask(mask, df, frame_idx):
    def write_mask(mask, flies):
        # flies should be list with entry for each fly,
        # each entry a 2 x 2 array; first axis time 2nd axis position

        palette = color_palette("Paired")
        color = lambda idx: tuple(color * 255 for color in palette[idx % len(palette)])

        for idx, position in enumerate(flies):
            mask = cv2.line(
                mask, tuple(position[0]), tuple(position[1]), color(idx), thickness=1,
            )
        return mask

    # Extracting local data
    local_df = df.query(f"{frame_idx - 1} <= frame <= {frame_idx}")

    # Only update if we have more than 1 position.
    if local_df["frame"].unique().shape[0] == 2:
        # Grouping per fly and putting it into right form for write_mask
        fly_locs = [
            fly_info[["x", "y"]].to_numpy(dtype=np.int32)
            for _, fly_info in local_df.groupby(by="ID")
        ]
        mask = write_mask(mask, fly_locs)
    return mask


def touching(image, df, frame_idx, touching_distance=15):
    """ Checks if each fly is within touching_distance of other fly"""
    local_locs = df.query(f"frame == {frame_idx}")[["x", "y"]].to_numpy()
    dist_matrix = distance_matrix(local_locs, local_locs)

    # minimum 2 cause of diagonal
    touching = np.sum(dist_matrix < touching_distance, axis=0) >= 2

    for idx in np.where(touching)[0]:
        image = cv2.circle(
            image,
            tuple(
                df.query(f"frame == {frame_idx} and ID == {idx}")[["x", "y"]]
                .to_numpy(dtype=np.int32)
                .squeeze()
            ),
            radius=5,
            color=(0, 0, 255),
            thickness=2,
        )

    return image
