import os
import pickle
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from PCutils import read_ply_files, encode_and_decode
from tqdm import tqdm
from sklearn.neighbors import KDTree
from parser_utils import parse_args

def get_labels_sample(
    i,
    kitti_folder,
    raw_folder,
    rec_folder,
    tmp_folder,
    codec_path,
    codec,
    quantization_parameters
):
    '''
    Compresses for various quantization parameters a single point cloud and
    performs recoloring, this is needed to obtain the mIoU plot that can be
    generated using the plot.py script
    '''
    sample_name = "".join(
        ["0" for i in range(6 - len(str(i)))]) + str(i) + ".npy"

    full_gt = np.load(
        os.path.join(
            kitti_folder,
            "gt",
            sample_name
        )
    )

    geom = np.load(
        os.path.join(
            kitti_folder,
            "velodyne",
            sample_name
        )
    )

    randla_pred = np.load(
        os.path.join(
            kitti_folder,
            "pred",
            sample_name
        )
    )

    reference_geom_class_file = os.path.join(
        raw_folder,
        f"geom_class{i}.ply"
    )

    rec_sample_file = os.path.join(
        rec_folder,
        f"reconstructed{i}.ply"
    )

    #building the KDtree that will later be used to compute the nearest neighbour
    tree = KDTree(geom, leaf_size=5)

    compressed_pcs = []
    gts = []
    randlas = []
    tmp_compressed_file = os.path.join(tmp_folder, f"compressed{i}.bin")
    for qp in quantization_parameters:

        rec_sample_file = os.path.join(
            tmp_folder,
            f"reconstructed_qp_{qp}_id_{i}.ply"
        )

        encode_and_decode(
            codec_path,
            reference_geom_class_file,
            tmp_compressed_file,
            rec_sample_file,
            quantization_bits=qp,
            codec=codec,
            pc_scale_factor=20,
            ascii_text=True,
            silence_output=True,
            encode_colors=False,
        )

        compressed_pc = read_ply_files(
            rec_sample_file,
            only_geom=True
        )

        #finding nearest neighbour for each point in the compressed PC
        _, nearest_neighbors = tree.query(compressed_pc[:, :3], k=1)
        gt = full_gt[nearest_neighbors.reshape(-1)]
        randla = randla_pred[nearest_neighbors.reshape(-1)]
        compressed_pcs.append(compressed_pc)
        gts.append(gt)
        randlas.append(randla)
    return compressed_pcs, gts, randlas




FLAGS = parse_args()
codecs_path = FLAGS.codecs_path

# definition of some useful path
codec = "tmc13"
if codec == "tmc13":
    codec_path = os.path.join(
        codecs_path,
        "mpeg-pcc-tmc13",
        "build",
        "tmc3",
        "tmc3"
    )
    quantization_parameters = range(5, -1, -1)
    quantization_parameters = range(6)
elif codec == "draco":
    codec_path = os.path.join(
        codecs_path,
        "draco",
        "build"
    )
    quantization_parameters = range(6, -1, -1)
else:
    raise Exception(f"NotImplementedError: {codec} codec not supported")

def miou(gt, pred):
    gt = gt[np.where(pred > 0)]
    pred = pred[np.where(pred > 0)]
    n_classes = max(np.max(gt), np.max(pred))
    metric = 0
    den = 0
    for i in range(1, n_classes):
        intersection = np.sum((gt == i) & (pred == i))
        union = np.sum((gt == i) | (pred == i))
        if union > 0:
            den += 1
            metric += intersection / union
    return metric/den

# folders
dataset_folder = os.path.join("..", "dataset")
kitti_folder = os.path.join(dataset_folder, "kitti")
raw_folder = os.path.join(dataset_folder, "raw")
rec_folder = os.path.join(dataset_folder, "rec")
tmp_folder = os.path.join("..", "dataset", "tmp")

# files
tmp_compressed_file = os.path.join(tmp_folder, "compressed.bin")

final_dict = {qp: {
    "geom": [],
    "gt": [],
    "CACTUS": [],
} for qp in quantization_parameters}

mious = []

get_labels_sample_partial = partial(
    get_labels_sample,
    kitti_folder=kitti_folder,
    raw_folder=raw_folder,
    rec_folder=rec_folder,
    tmp_folder=tmp_folder,
    codec_path=codec_path,
    codec=codec,
    quantization_parameters=quantization_parameters
)
with Pool(cpu_count(), maxtasksperchild=1) as p:
    return_data = list(tqdm(p.imap(get_labels_sample_partial, range(100))))

for compressed_pcs, gts, randlas in return_data:
    for i, qp in enumerate(quantization_parameters):
        final_dict[qp]["geom"].append(compressed_pcs[i])
        final_dict[qp]["gt"].append(gts[i])
        final_dict[qp]["CACTUS"].append(randlas[i])

with open("../dataset/kitti/data.pickle", "wb") as f:
    pickle.dump(final_dict, f)
