import os
import h5py
import analysis_utils

DATA_DIR = ""
MODEL_DIR = r"E:\sam\motionmapperpy\data\invivo3cam2\UMAP"
N_GROUPS = [20, 36, 149]

for N in N_GROUPS:
    wshed_fname = os.path.join(MODEL_DIR, f"zVals_wShed_groups{N}.mat")

    wshed_file = h5py.File(wshed_fname, "r")
    labels = wshed_file["watershedRegions"]

    bouts = analysis_utils.get_bout_lengths(labels)
    analysis_utils.plot_bout_lengths_together(
        bouts, os.path.join(MODEL_DIR, f"bout_lengths_{N}motifs.png")
    )
# analysis_utils.plot_bout_lengths(bouts, 7, 3, cluster_0index=False)

# analysis_utils.plot_motif_usage(
#     labels, os.path.join(MODEL_DIR, f"motif_usage{N_GROUPS}.png")
# )

# analysis_utils.plot_cluster_examples(
#     labels, 3, os.path.join(MODEL_DIR, f"wshed{N_GROUPS}"), raw=True
# )

wshed_file.close()
