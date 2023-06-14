import json
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.neighbors import KDTree
from PCutils import write_ply_file, encode_and_decode
from PCutils import read_ply_files, bj_delta, D1_PSNR
from parser_utils import parse_args
from segmentation_codec import encoder, decoder

def assign_closest_class(pc1, pc2, data):

    '''
    Parameters:
        pc1 (np.ndarray): geometry of the full point cloud,
                          must have shape (n, 3) where n is the number of points
        pc2 (np.ndarray): geometry of the compressed point cloud,
                          must have shape (n, 3) where n is the number of points
                     scale factor is chosen as the maximum coordinate excursion
                     of pc1
        data (np.ndarray): classes for the first point cloud
    Returns (int): the second point cloud with class information
    '''

    # finding out distance between points in pc2 and
    # their nn in pc1
    tree1 = KDTree(pc1, leaf_size=32)
    indexes = tree1.query(pc2, k=1, return_distance=False)
    return np.concatenate([pc2, data[indexes[:, 0]]], axis=1)

def full_compress(
    i,
    codec,
    kitti_folder,
    raw_folder,
    compressed_folder,
    rec_folder,
    tmp_folder,
    codec_path,
    quantization_parameters,
    config_values,
):
    '''
    Performs compression of a pc both with standard codec and with cactus and also analyzes
    performance on the partitioned pc
    '''
    time_single = 0
    time_CACTUS = 0
    n_steps = 0

    ## path to temporary intermediate compressions
    tmp_compressed_file = os.path.join(tmp_folder, f"compressed{i}.bin")
    ## path to temporary reconstructed files
    tmp_rec_file = os.path.join(tmp_folder, f"reconstructed{i}.ply")
    ## path to the compressed file
    compressed_file = os.path.join(compressed_folder, f"compressed{i}.bin")
    ## path to the reconstructed pc
    reconstructed_file = os.path.join(rec_folder, f"rec_clustered{i}.ply")

    sample_name = "".join(
        ["0" for i in range(6 - len(str(i)))]) + str(i) + ".npy"
    plottable_data = {
        f"{codec}_geom_size": [],
        f"{codec}_geom_class_size": [],
        "CACTUS_size": [],
        f"{codec}_geom_PSNR": [],
        f"{codec}_geom_class_PSNR": [],
        "CACTUS_PSNR": [],
        "standard_avg_PSNR": [],
        "CACTUS_avg_PSNR": [],
    }

    # loading the points and the classes attributes for pc i
    geom = np.load(
        os.path.join(
            kitti_folder,
            "velodyne",
            sample_name
        )
    )
    point_class = np.load(
        os.path.join(
            kitti_folder,
            "pred",
            sample_name
        )
    )

    # scale used by the evaluate script
    scale = np.max(np.max(geom, axis=0) - np.min(geom, axis=0))

    # file that will contain both geometry and classes for the pc
    reference_geom_class_file = os.path.join(
        raw_folder,
        f"geom_class{i}.ply"
    )

    # file that will contain only geometry of the pc
    reference_geom_file = os.path.join(
        raw_folder,
        f"geom{i}.ply"
    )

    # creating the ply file with geometry and classes
    write_ply_file(
        geom,
        reference_geom_class_file,
        ascii_text=True,
        attributes=np.hstack([point_class, point_class, point_class]),
        dtype=["uint8", "uint8", "uint8"],
        names=["red", "green", "blue"]
    )

    # creating the ply file with geometry
    write_ply_file(
        geom,
        reference_geom_file,
        ascii_text=True
    )

    # path to the file compressed with the standard codec
    compressed_sample_file = os.path.join(
        compressed_folder,
        f"compressed{i}.bin"
    )


    # path to the file reconstructed with the standard codec
    rec_sample_file = os.path.join(
        rec_folder,
        f"reconstructed{i}.ply"
    )

    #lists where to save information to be plotted
    average_psnrs_standard = []
    average_psnrs_CACTUS = []

    # bool variable that detects problems with encoding/decoding and skips
    # the current sample
    can_continue = True

    for key in plottable_data:
        plottable_data[key].append([])

    #for each quantization parameter
    for qp in quantization_parameters:
        n_steps += 1

        #reduces part of the config values
        for key in range(20):
            config_values[str(key)] = qp

        # encoding and decoding the pc (also with classes) using the standard codec
        time_single += encode_and_decode(
            codec_path,
            reference_geom_class_file,
            tmp_compressed_file,
            tmp_rec_file,
            quantization_bits=qp,
            codec=codec,
            pc_scale_factor=20,
            ascii_text=True,
            silence_output=True,
            encode_colors=True
        )

        # this code is needed to avoid recoloring since by default tmc13
        # averages the attributes of the nearest neighbors to the point
        # which doesn't make sense if the attribute is a class
        if codec == "tmc13":
            temp_comp_geom = read_ply_files(tmp_rec_file, only_geom=False)
            temp_pc = assign_closest_class(
                geom,
                temp_comp_geom[:, :3],
                point_class
            )
            tmp_point_class = temp_pc[:, 3:]

            write_ply_file(
                temp_comp_geom.astype(np.float32),
                tmp_rec_file,
                ascii_text=True,
                attributes=np.hstack([
                    tmp_point_class,
                    tmp_point_class,
                    tmp_point_class
                ]),
                dtype=["uint8", "uint8", "uint8"],
                names=["red", "green", "blue"]
            )

            encode_and_decode(
                codec_path,
                tmp_rec_file,
                tmp_compressed_file,
                tmp_rec_file,
                quantization_bits=0,
                codec=codec,
                # the extra 0.00001 is needed because there is some bug in TMC13
                # that makes the coding lossy even though it should be lossless
                # with codingScale = 1 and scale factor 20 (in this case).
                # We tested eveything and the extra 0.00001 does not affect the
                # decoded classes and geometry that are correct and it solves
                # the aformentioned bug even though it is just a hack
                pc_scale_factor=20.00001,
                ascii_text=True,
                silence_output=True,
                encode_colors=True,
                print_command=False
            )

        # reading the reconstructed pc
        compressed_class_points = read_ply_files(
            tmp_rec_file,
            only_geom=False
        )
        if codec == "tmc13":
            class_diff = np.sum(temp_pc[:, 3] != compressed_class_points[:, 3])
            if class_diff > 0:
                print("Here there is a class error")
            geom_diff = np.abs(temp_pc[:, :3] - \
                    compressed_class_points[:, :3]).sum()
            if geom_diff > 0:
                print("Here there is a geometry error")

        # encoding and decoding the pc using the standard codec
        encode_and_decode(
            codec_path,
            reference_geom_file,
            compressed_sample_file,
            rec_sample_file,
            quantization_bits=qp,
            codec=codec,
            pc_scale_factor=20,
            silence_output=True
        )

        standard_size = os.stat(compressed_sample_file).st_size

        quantizations = qp

        #encoding with CACTUS
        tot_time = encoder(
            geom,
            point_class,
            compressed_file,
            tmp_folder,
            codec_path,
            codec=codec,
            config_values=quantizations,
            index=i
        )
        time_CACTUS += tot_time

        #decoding with CACTUS
        _, can_continue, tot_time = decoder(
            compressed_file,
            reconstructed_file,
            tmp_folder,
            codec_path,
            codec,
            index=i
        )
        time_CACTUS += tot_time

        size = os.stat(compressed_file).st_size

        # computing the various PSNRs
        if can_continue:

            # getting the reconstructed classes
            class_size = os.stat(tmp_compressed_file).st_size
            plottable_data[f"{codec}_geom_class_size"][-1].append(class_size)

            # reading the PC reconstructed with CACTUS
            CACTUS_rec_pc = read_ply_files(
                reconstructed_file,
                only_geom=False,
                att_name="class"
            )

            considerable_classes = np.unique(CACTUS_rec_pc[:, 3])
            average_psnr_standard = 0
            average_psnr_CACTUS = 0
            n_sums = 0
            for considerable_class in considerable_classes:
                semantic_CACTUS = CACTUS_rec_pc[
                    np.where(CACTUS_rec_pc[:, 3] == considerable_class)
                ][:, :3]

                semantic_standard = compressed_class_points[
                    np.where(
                        compressed_class_points[:, 3] == considerable_class
                    )
                ][:, :3]
                semantic_correct = geom[
                    np.where(point_class[:, 0] == considerable_class)
                ]
                # arbitrarily chose that the class should have more than 5 points to be meaningful
                # otherwise CACTUS can get 0 MSE leading to an unfair advantage in the average that is
                # only due to its ability to reconstruct very well small pcs
                if len(semantic_standard) > 5 and len(semantic_CACTUS) > 5:
                    average_psnr_CACTUS += D1_PSNR(
                        semantic_CACTUS,
                        semantic_correct,
                        scale
                    )

                    average_psnr_standard += D1_PSNR(
                        semantic_standard,
                        semantic_correct,
                        scale
                    )
                    n_sums += 1
            average_psnrs_CACTUS.append(average_psnr_CACTUS / n_sums)
            average_psnrs_standard.append(average_psnr_standard / n_sums)

            # computing psnr for CACTUS
            psnr_seg = D1_PSNR(
                CACTUS_rec_pc[:, :3],
                geom,
                scale
            )
            # computing psnr for the standard codec
            psnr_standard = D1_PSNR(
                read_ply_files(rec_sample_file, only_geom=True),
                geom,
                scale
            )

            plottable_data[f"{codec}_geom_size"][-1].append(standard_size)
            plottable_data["CACTUS_size"][-1].append(size)
            plottable_data[f"{codec}_geom_PSNR"][-1].append(psnr_standard)
            plottable_data["CACTUS_PSNR"][-1].append(psnr_seg)
            plottable_data[f"{codec}_geom_class_PSNR"][-1].append(psnr_standard)
            plottable_data["CACTUS_avg_PSNR"][-1].append(
                average_psnr_CACTUS / n_sums
            )
            plottable_data["standard_avg_PSNR"][-1].append(
                average_psnr_standard / n_sums
            )
        else:
            # if one error happened during decoding the results
            # from the considered sample are removed
            for key in plottable_data:
                plottable_data[key].pop()

            break

    if can_continue:

        delta_rate_s_ss = bj_delta(
            plottable_data[f"{codec}_geom_size"][-1],
            plottable_data[f"{codec}_geom_PSNR"][-1],
            plottable_data["CACTUS_size"][-1],
            plottable_data["CACTUS_PSNR"][-1],
            mode=1
        )

        delta_PSNR_s_ss = bj_delta(
            plottable_data[f"{codec}_geom_size"][-1],
            plottable_data[f"{codec}_geom_PSNR"][-1],
            plottable_data["CACTUS_size"][-1],
            plottable_data["CACTUS_PSNR"][-1],
            mode=0
        )

        delta_rate_ss_c = bj_delta(
            plottable_data[f"{codec}_geom_class_size"][-1],
            plottable_data[f"{codec}_geom_class_PSNR"][-1],
            plottable_data["CACTUS_size"][-1],
            plottable_data["CACTUS_PSNR"][-1],
            mode=1
        )

        delta_PSNR_ss_c = bj_delta(
            plottable_data[f"{codec}_geom_class_size"][-1],
            plottable_data[f"{codec}_geom_class_PSNR"][-1],
            plottable_data["CACTUS_size"][-1],
            plottable_data["CACTUS_PSNR"][-1],
            mode=0
        )

        delta_psnr_average = bj_delta(
            np.mean(
                plottable_data[f"{codec.lower()}_geom_size"], axis=0
            ) / 1024,
            average_psnrs_standard,
            np.mean(plottable_data["CACTUS_size"], axis=0) / 1024,
            average_psnrs_CACTUS,
            mode=0
        )

        delta_rate_average = bj_delta(
            np.mean(
                plottable_data[f"{codec.lower()}_geom_size"], axis=0
            ) / 1024,
            average_psnrs_standard,
            np.mean(plottable_data["CACTUS_size"], axis=0) / 1024,
            average_psnrs_CACTUS,
            mode=1
        )

        bj_deltas = [
            delta_rate_s_ss,
            delta_PSNR_s_ss,
            delta_rate_ss_c,
            delta_PSNR_ss_c,
            delta_rate_average,
            delta_psnr_average
        ]
        return plottable_data, bj_deltas, time_single, time_CACTUS, n_steps
    else:
        print("There was some kind of error while encoding or decoding the pc")
        return None

FLAGS = parse_args()
codec = FLAGS.codec.lower()
config_file = FLAGS.config_file
codecs_path = FLAGS.codecs_path

bj_deltas = []

with open(config_file, "r") as f:
    config_values = json.load(f)

# defining the paths to the codecs executables (in this case TMC13 or draco)
if codec == "tmc13":
    codec_path = os.path.join(
        codecs_path,
        "mpeg-pcc-tmc13",
        "build",
        "tmc3",
        "tmc3"
    )
    quantization_parameters = range(5, -1, -1)
elif codec == "draco":
    codec_path = os.path.join(
        codecs_path,
        "draco",
        "build"
    )
    quantization_parameters = range(10, -1, -2)
else:
    raise Exception(f"NotImplementedError: {codec} codec not supported")


# definition of some useful path

# folders
## folder that contains all the data, raw and processed
dataset_folder = os.path.join("..", "dataset")
## folder that contains data from the kitti dataset
kitti_folder = os.path.join(dataset_folder, "kitti")
## folder containing the uncompressed pcs in ply format
raw_folder = os.path.join(dataset_folder, "raw")
## folder containing the compressed point clouds (bin files)
compressed_folder = os.path.join(dataset_folder, "compressed")
## folder containing the reconstructed ply files
rec_folder = os.path.join(dataset_folder, "rec")
## folder for temporary stuff
tmp_folder = os.path.join("..", "dataset", "tmp")

# variable containing data that can be plotted when considering
# the coding of the full PC
plottable_data = {
    f"{codec}_geom_size": [],
    f"{codec}_geom_class_size": [],
    "CACTUS_size": [],
    f"{codec}_geom_PSNR": [],
    f"{codec}_geom_class_PSNR": [],
    "CACTUS_PSNR": [],
    "standard_avg_PSNR": [],
    "CACTUS_avg_PSNR": [],
}

# variables needed to compute compression time
time_single = 0
time_CACTUS = 0
n_steps = 0

# considering only the first 100 samples since it is actually quite slow to process
# the whole dataset since there is a lot of time overhead due to the fact that
# in order to interact with the codecs the PCs need to be read and written in memory
full_compress_partial = partial(
    full_compress,
    codec=codec,
    kitti_folder=kitti_folder,
    raw_folder=raw_folder,
    compressed_folder=compressed_folder,
    rec_folder=rec_folder,
    tmp_folder=tmp_folder,
    codec_path=codec_path,
    quantization_parameters=quantization_parameters,
    config_values=config_values,
)
with Pool(cpu_count(), maxtasksperchild=1) as p:
    return_data = list(tqdm(p.imap(full_compress_partial, range(100))))
for key in plottable_data:
    plottable_data[key] = []
for return_value in return_data:
    data_dict, bj_deltas_i, time_single_i, time_CACTUS_i, n_steps_i = return_value
    for key in data_dict:
        plottable_data[key] += data_dict[key]

    bj_deltas.append(bj_deltas_i)
    time_single += time_single_i
    time_CACTUS += time_CACTUS_i
    n_steps += n_steps_i

print(f"Total time for normal codec: {time_single / n_steps}, "
        f"Total time for CACTUS: {time_CACTUS / n_steps}")

if codec == "draco":
    codec = "Draco"

if codec.lower() == "tmc13":
    codec = "TMC13"

print("Min", np.min(bj_deltas, axis=0))
print("Max", np.max(bj_deltas, axis=0))
mean_bj = np.mean(bj_deltas, axis=0)
print(f"DELTA RATE (Codec(geom), CACTUS): {mean_bj[0]}")
print(f"DELTA PSNR (Codec(geom), CACTUS): {mean_bj[1]}")
print(f"DELTA RATE (Codec(geom + class), CACTUS): {mean_bj[2]}")
print(f"DELTA PSNR (Codec(geom + class), CACTUS): {mean_bj[3]}")
print(f"DELTA RATE (Codec avg, CACTUS avg): {mean_bj[4]}")
print(f"DELTA PSNR (Codec avg, CACTUS avg): {mean_bj[5]}")
plt.figure(figsize=(6.4, 4))
matplotlib.rcParams.update({"font.size": 14})
plt.plot(
    np.mean(plottable_data[f"{codec.lower()}_geom_class_size"], axis=0) / 1024,
    np.mean(plottable_data[f"{codec.lower()}_geom_class_PSNR"], axis=0),
    "--.",
    label=f"{codec} with classes"
)
plt.plot(
    np.mean(plottable_data[f"{codec.lower()}_geom_size"], axis=0) / 1024,
    np.mean(plottable_data[f"{codec.lower()}_geom_PSNR"], axis=0),
    ".-",
    label=f"{codec} wo classes"
)
plt.plot(
    np.mean(plottable_data["CACTUS_size"], axis=0) / 1024,
    np.mean(plottable_data["CACTUS_PSNR"], axis=0),
    "-..",
    label="CACTUS"
)
plt.xlabel("Compressed Size (kB)")
plt.ylabel("PSNR(dB)")
leg = plt.legend()
plt.tight_layout()

plt.savefig(f"../figures/{codec}_with_ss.png", bbox_inches='tight')
plt.savefig(f"../figures/{codec}_with_ss.pdf", bbox_inches='tight')

plt.figure()
plt.plot(
    np.mean(plottable_data["CACTUS_size"], axis=0) / 1024,
    np.mean(plottable_data["CACTUS_avg_PSNR"], axis=0),
    "-..",
    label=f"CACTUS {codec} average",
    color="green"
)
plt.plot(
    np.mean(plottable_data[f"{codec.lower()}_geom_size"], axis=0) / 1024,
    np.mean(plottable_data["standard_avg_PSNR"], axis=0),
    ".-",
    label=f"{codec} average",
    color="darkorange"
)
plt.xlabel("Compressed Size (kB)")
plt.ylabel("Average PSNR(dB)")
leg = plt.legend()
plt.tight_layout()
results_dict = {
    f"size_CACTUS_{codec}": list(
        np.mean(plottable_data["CACTUS_size"], axis=0) / 1024
    ),
    f"PSNR_CACTUS_{codec}": list(np.mean(plottable_data["CACTUS_avg_PSNR"], axis=0)),
    f"size_{codec}": list(
        np.mean(plottable_data[f"{codec.lower()}_geom_size"], axis=0) / 1024
    ),
    f"PSNR_{codec}": list(np.mean(plottable_data["standard_avg_PSNR"], axis=0)),
}
if os.path.exists("plot_data.json"):
    with open("plot_data.json", "r") as f:
        results_dict.update(json.load(f))
with open("plot_data.json", "w") as f:
    json.dump(results_dict, f, indent=4)

plt.savefig(f"../figures/{codec}_average.png", bbox_inches='tight')
plt.savefig(f"../figures/{codec}_average.pdf", bbox_inches='tight')
plt.show()
