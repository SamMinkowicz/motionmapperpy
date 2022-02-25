import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def load_params(model_dir):
    # load training params
    params = None
    for f in os.scandir(model_dir):
        if f.name.startswith("params"):
            with open(f.path, "rb") as params_f:
                params = pickle.load(params_f)

    return params


def get_bout_lengths(labels):
    """
    Let a bout be some number of consecutive frames with the same cluster.
    Count the length of each bout for each cluster.
    Return a dictionary mapping clusters to a list of bout lengths
    """
    counts = {label: [] for label in np.unique(labels)}
    for frame, label in enumerate(np.squeeze(labels)):
        if frame == 0:
            previous_label = label
            count = 1
            continue

        if label == previous_label:
            count += 1
        else:
            counts[previous_label].append(count)
            previous_label = label
            count = 1

    counts[label].append(count)

    return counts


def get_bout_frames(labels):
    """
    Let a bout be some number of consecutive frames with the same cluster.
    Count the length of each bout for each cluster.
    Return dictionary mapping clusters to a list of lists of
    start and end frame indices for each bout
    """
    frames = {label: [] for label in np.unique(labels)}
    for frame, label in enumerate(np.squeeze(labels)):
        if frame == 0:
            previous_label = label
            frames[label].append([frame])
            continue

        if label != previous_label:
            frames[previous_label][-1].append(frame - 1)
            frames[label].append([frame])
            previous_label = label

    frames[label][-1].append(frame)

    return frames


def plot_bout_lengths_together(bouts, out_filename=None):
    """
    plot distribution of bout lengths. Plots bout lengths for all clusters together
    and then each separate.
    """
    # plot distribution of cluster bout lengths
    all_bout_lengths = np.hstack(bouts.values())
    plt.hist(all_bout_lengths, bins=500)
    plt.title("all bout lengths")
    plt.xlabel("Bout length")
    plt.ylim((0, 2500))
    if out_filename:
        plt.savefig(out_filename)
        plt.close()
    plt.show()


def plot_bout_lengths(bouts, n_rows, n_cols, cluster_0index=True):
    """
    plot distribution of bout lengths. Plots bout lengths for all clusters together
    and then each separate.
    """
    # plot distribution of cluster bout lengths
    # all_bout_lengths = np.hstack(bouts.values())
    # plt.hist(all_bout_lengths, bins=500)
    # plt.title("all bout lengths")
    # plt.xlabel("Bout length")
    # plt.show()

    # plot distribution of cluster bout lengths separately for each cluster
    cluster = 0
    # the watershed clusters start at 1
    if not cluster_0index:
        cluster = 1
    n_clusters = len(bouts.keys()) - cluster

    fig, axs = plt.subplots(n_rows, n_cols)
    for row in range(n_rows):
        for col in range(n_cols):
            if cluster > n_clusters:
                break
            axs[row, col].hist(bouts[cluster], bins=50)
            cluster += 1
    plt.tight_layout()
    plt.show()


def plot_motif_usage(labels, out_filename=None, sorted=True):
    """plot the usage % for each motif"""
    labels = np.squeeze(labels)
    unique_labels = np.unique(labels)
    total_frames = labels.size
    usage = [
        100 * np.round(np.count_nonzero(labels == label) / total_frames, 2)
        for label in unique_labels
    ]
    usage = np.array(usage)
    if sorted:
        usage = usage[np.argsort(-usage)]
    plt.plot(usage)
    plt.xlabel("Motif")
    plt.ylabel("Usage percent")

    if out_filename:
        plt.savefig(out_filename)
        plt.close()
    else:
        plt.show()


def plot_cluster_example(poses, limb_dict, out_path):
    """
    save an animated plot of a bout
    poses will be in shape limbs x coordinates x time
    """
    connected = [
        ("snout", "l_eye"),
        ("l_eye", "r_eye"),
        ("snout", "r_eye"),
        ("l_eye", "l_ear"),
        ("r_eye", "r_ear"),
        ("l_paw", "l_wrist"),
        ("l_wrist", "l_elbow"),
        ("l_elbow", "l_shoulder"),
        ("l_shoulder", "l_hindpaw"),
        ("r_paw", "r_wrist"),
        ("r_wrist", "r_elbow"),
        ("r_elbow", "r_shoulder"),
        ("r_shoulder", "r_hindpaw"),
    ]

    class PoseAnimation:
        def __init__(self, ax, limits, n_frames) -> None:
            self.ax = ax

            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

            self.ax.set_xlim(limits[:2])
            self.ax.set_ylim(limits[2:4])
            self.ax.set_zlim(limits[4:])
            self.ax.set_title(f"{n_frames} frames")

            self.lines = [
                self.ax.plot([], [], [], color=f"C{i}", linewidth=3)[0]
                for i, pair in enumerate(connected)
            ]

        def __call__(self, j):
            for line, pair in zip(self.lines, connected):
                line.set_data(
                    [poses[limb_dict[pair[i]], 0, j] for i in range(2)],
                    [poses[limb_dict[pair[i]], 1, j] for i in range(2)],
                )
                line.set_3d_properties(
                    [poses[limb_dict[pair[i]], 2, j] for i in range(2)],
                )
            return self.lines

    n_frames = poses.shape[2]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    limits = [
        np.min(poses[:, 0]),
        np.max(poses[:, 0]),
        np.min(poses[:, 1]),
        np.max(poses[:, 1]),
        np.min(poses[:, 2]),
        np.max(poses[:, 2]),
    ]

    pose_anim = PoseAnimation(ax, limits, n_frames)
    anim = animation.FuncAnimation(fig, pose_anim, frames=n_frames, blit=True)
    anim.save(out_path, animation.PillowWriter(fps=30))
    plt.close()


def plot_cluster_examples(labels, n_plots, out_dir, raw=True, min_frames=None):
    """plot n example animations of each cluster"""
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # load the pose data
    data_dir = r"E:\sam\anipose_out"
    if raw:
        data_dir = os.path.join(data_dir, "raw")

    limb_dict = None
    pose_file_prefix = "pose" if not raw else "pose_raw"
    all_poses = []
    for f in os.scandir(data_dir):
        if f.name.endswith("npy"):
            limbs_file = f.name.replace("npy", "pickle")
            limbs_file = limbs_file.replace(pose_file_prefix, "limbs")
            limbs_path = os.path.join(data_dir, limbs_file)

            if not os.path.exists(limbs_path):
                continue

            all_poses.append(np.load(f.path))

            # only need to get this once
            if limb_dict:
                continue

            with open(limbs_path, "rb") as f:
                limb_dict = pickle.load(f)

    # this is now limbs x coordinates x frames
    poses = np.dstack(all_poses)
    all_bout_frames = get_bout_frames(labels)

    for cluster in np.unique(labels):
        print(f"cluster: {cluster}")
        bout = 0
        while bout < n_plots:
            # for bout in range(n_plots):

            # check whether there are this many bouts for the given cluster
            if bout >= len(all_bout_frames[cluster]):
                break

            # get the pose data for the given bout
            bout_frames = all_bout_frames[cluster][bout]

            if isinstance(min_frames, int):
                if bout_frames[1] - bout_frames[0] < min_frames:
                    bout += 1
                    continue
            print(f"bout: {bout}")
            bout_pose = poses[:, :, bout_frames[0] : bout_frames[1]]

            # pass the data and out_filename to plot_cluster_example
            gif_filename = f"anim_cluster{cluster}_bout{bout}.gif"
            if raw:
                gif_filename = gif_filename.replace(".gif", "_raw.gif")
            plot_cluster_example(
                bout_pose, limb_dict, os.path.join(out_dir, gif_filename)
            )
            bout += 1


# plot_cluster_examples(
#     get_gmm_clusters(MODEL_DIR),
#     5,
#     os.path.join(MODEL_DIR, "gmm_cluster_gifs"),
#     raw=True,
# )
# plot_cluster_examples(
#     get_gmm_clusters(MODEL_DIR),
#     5,
#     os.path.join(MODEL_DIR, "gmm_cluster_gifs"),
#     raw=False,
# )


# def plot_all_bout_lengths():
#     gmm_labels = get_gmm_clusters(MODEL_DIR)
#     bouts = get_bout_lengths(gmm_labels)
#     plot_bout_lengths(bouts, 11, 5)

#     kneed_merged = load_gmm_results(MODEL_DIR, "kneed")["merged_labels"]
#     kneed_bouts = get_bout_lengths(kneed_merged)
#     plot_bout_lengths(kneed_bouts, 6, 4)

#     piece_merged = load_gmm_results(MODEL_DIR, "piecewise")["merged_labels"]
#     piece_bouts = get_bout_lengths(piece_merged)
#     plot_bout_lengths(piece_bouts, 8, 5)


# def plot_all_usage():
#     """
#     plot the usage percent for each motif for the gmm labels
#     and for the labels after the two types of merging
#     """
#     gmm_labels = get_gmm_clusters(MODEL_DIR)
#     kneed_merged = load_gmm_results(MODEL_DIR, "kneed")["merged_labels"]
#     piece_merged = load_gmm_results(MODEL_DIR, "piecewise")["merged_labels"]
#     labels = [gmm_labels, kneed_merged, piece_merged]
#     filenames = [
#         os.path.join(MODEL_DIR, filename)
#         for filename in [
#             "gmm_labels_usage",
#             "kneed_labels_usage",
#             "piecewise_labels_usage",
#         ]
#     ]

#     for i in range(3):
#         plot_motif_usage(labels[i], filenames[i])

