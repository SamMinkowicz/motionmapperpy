import glob, os, pickle
from datetime import datetime
import os

import numpy as np
from scipy.io import loadmat, savemat
import hdf5storage
import sys
from utils_sm import load_training_data, save_hyperparams

sys.path.append("/mnt/HFSP_Data/scripts/motionmapperpy")
import motionmapperpy as mmpy


"""1. load data"""
local_dir = r"E:/sam/motionmapperpy"

# Create a project folder which contains all the data that you want to embed in a single map.
projectPath = os.path.join(local_dir, "data/invivo3cam2")
mmpy.createProjectDirectory(projectPath)

data = load_training_data(r"E:/sam/anipose_out", trim=False)

print(data.shape)
savemat(
    os.path.join(projectPath, "Projections/dataset_0_pcaModes.mat"),
    {"projections": data},
)

"""2. Setup run parameters for MotionMapper."""

parameters = mmpy.setRunParameters()

# %%%%%%% PARAMETERS TO CHANGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.projectPath = projectPath
parameters.method = "UMAP"

parameters.waveletDecomp = (
    True  #% Whether to do wavelet decomposition. If False, PCA projections are used for
)
#% tSNE embedding.

parameters.minF = 0.5  #% Minimum frequency for Morlet Wavelet Transform

parameters.maxF = 62  #% Maximum frequency for Morlet Wavelet Transform,
#% equal to Nyquist frequency for your measurements.

parameters.samplingFreq = 125  #% Sampling frequency (or FPS) of data.

parameters.numPeriods = 25  #% No. of frequencies between minF and maxF.

parameters.numProcessors = -1  #% No. of processor to use when parallel
#% processing (for wavelets, if not using GPU). -1 to use all cores.

parameters.useGPU = 1  # GPU to use, set to -1 if GPU not present

parameters.training_numPoints = 2000  #% Number of points in mini-tSNEs.

# %%%%% NO NEED TO CHANGE THESE UNLESS RAM (NOT GPU) MEMORY ERRORS RAISED%%%%%%%%%%
parameters.trainingSetSize = 1000  #% Total number of representative points to find. Increase or decrease based on
#% available RAM. For reference, 36k is a good number with 64GB RAM.

parameters.embedding_batchSize = (
    30000  #% Lower this if you get a memory error when re-embedding points on learned
)
#% tSNE map.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


projectionFiles = glob.glob(parameters.projectPath + "/Projections/*pcaModes.mat")
for i in projectionFiles:
    print(i)

m = loadmat(projectionFiles[0], variable_names=["projections"])["projections"]

# %%%%%
parameters.pcaModes = m.shape[1]  #%Number of PCA projections in saved files.
parameters.numProjections = parameters.pcaModes
# %%%%%
del m

print(datetime.now().strftime("%m-%d-%Y_%H-%M"))
print("tsneStarted")

if parameters.method == "TSNE":
    if parameters.waveletDecomp:
        tsnefolder = parameters.projectPath + "/TSNE/"
    else:
        tsnefolder = parameters.projectPath + "/TSNE_Projections/"
elif parameters.method == "UMAP":
    tsnefolder = parameters.projectPath + "/UMAP/"

if not os.path.exists(tsnefolder + "training_embedding.mat"):
    print("Running minitSNE")
    mmpy.subsampled_tsne_from_projections(parameters, parameters.projectPath)
    print("minitSNE done, finding embeddings now.")
    print(datetime.now().strftime("%m-%d-%Y_%H-%M"))

import h5py

with h5py.File(tsnefolder + "training_data.mat", "r") as hfile:
    trainingSetData = hfile["trainingSetData"][:].T

with h5py.File(tsnefolder + "training_embedding.mat", "r") as hfile:
    trainingEmbedding = hfile["trainingEmbedding"][:].T

if parameters.method == "TSNE":
    zValstr = "zVals" if parameters.waveletDecomp else "zValsProjs"
else:
    zValstr = "uVals"

for i in range(len(projectionFiles)):
    print("Finding Embeddings")
    print("%i/%i : %s" % (i + 1, len(projectionFiles), projectionFiles[i]))
    if os.path.exists(projectionFiles[i][:-4] + "_%s.mat" % (zValstr)):
        print("Already done. Skipping.\n")
        continue

    projections = loadmat(projectionFiles[i])["projections"]
    zValues, outputStatistics = mmpy.findEmbeddings(
        projections, trainingSetData, trainingEmbedding, parameters
    )

    hdf5storage.write(
        data={"zValues": zValues},
        path="/",
        truncate_existing=True,
        filename=projectionFiles[i][:-4] + "_%s.mat" % (zValstr),
        store_python_metadata=False,
        matlab_compatible=True,
    )
    with open(
        projectionFiles[i][:-4] + "_%s_outputStatistics.pkl" % (zValstr), "wb"
    ) as hfile:
        pickle.dump(outputStatistics, hfile)

    print("Embeddings saved.\n")
    del zValues, projections, outputStatistics

print("All Embeddings Saved!")

mmpy.findWatershedRegions(
    parameters,
    minimum_regions=40,
    startsigma=0.3,
    pThreshold=[0.33, 0.67],
    saveplot=True,
    endident="*_pcaModes.mat",
)


save_hyperparams(os.path.join(projectPath, "params.pkl"), parameters)
