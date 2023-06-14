import os
from time import time
import numpy as np
from PCutils import write_ply_file, encode
from PCutils import read_ply_files, decode


def encoder(
    points,
    point_class,
    compressed_path,
    temp_folder,
    codec_path,
    codec="draco",
    config_values=0,
    index=0
):
    '''
    CACTUS encoder function
    Parameters:
        points (np.ndarray): coordinates of the points, its shape should be (npts, 3)
        point_class (np.ndarray): classes of the points
        compressed_path (str): path to the compressed represenation using CACTUS
        temp_folder (str): path to a folder where to save intermediate files
        codec_path (str): path to the codec
        codec (str): codec used, can be either draco or tmc13
        config_values (list): qps for the various classes
        index (int): unique identifier for the pc (needed when parallelizing to avoid overwriting
            compressed represenations of other pcs)
    '''
    total_time = 0
    all_points = np.hstack([
        points[:, :3],
        point_class.reshape(-1, 1)
    ])
    broken_classes = []

    for cluster_id in np.unique(point_class):

        start_time = time()
        if isinstance(config_values, int):
            quantization_bits = config_values
        else:
            quantization_bits = config_values[str(cluster_id)]

        segmented_points = all_points[
            np.where(
                all_points[:, 3].reshape(-1) == cluster_id
            )
        ]
        total_time += time() - start_time

        if segmented_points.shape[0] > 0 and quantization_bits >= 0:

            segmented_name = os.path.join(
                temp_folder,
                f"seg_class_{cluster_id}_{index}.ply"
            )

            write_ply_file(segmented_points, segmented_name)

            comp_pc_geom = os.path.join(
                temp_folder,
                f"segment_{cluster_id}_{index}.bin"
            )

            total_time += encode(
                codec_path,
                segmented_name,
                comp_pc_geom,
                quantization_bits=quantization_bits,
                codec=codec,
                pc_scale_factor=20,
            )

            if codec.lower() == "draco":
                possibly_broken_pc = f"../dataset/tmp/maybe_broken_{index}.ply"
                decode(
                    comp_pc_geom,
                    possibly_broken_pc,
                    codec_path,
                    codec=codec,
                )
                broken_pc = read_ply_files(possibly_broken_pc, only_geom=True)

                if max(np.max(broken_pc), -np.min(broken_pc)) > 1e20:
                    broken_classes.append(cluster_id)

    start_time = time()
    with open(compressed_path, "wb") as f:
        covered_classes = np.unique(point_class)
        covered_classes = set(covered_classes).difference(set(broken_classes))
        if isinstance(config_values, int):
            header = "".join(
                ["0" if x not in covered_classes else "1" for x in range(32)])
        else:
            header = "".join(
                ["0" if (x not in covered_classes) or (config_values[str(x)] < 0) \
                        else "1" for x in range(32)])
        content = int(header, 2).to_bytes(4, "little")
        for cluster_id in covered_classes:

            comp_data_path = os.path.join(
                temp_folder,
                f"segment_{cluster_id}_{index}.bin"
            )

            with open(comp_data_path, "rb") as temp_f:
                content += os.stat(comp_data_path).st_size.to_bytes(4, "little")
                content += temp_f.read()
        f.write(content)
    total_time += time() - start_time
    return total_time

def decoder(
    compressed_path,
    reconstructed_path,
    temp_folder,
    codec_path,
    codec,
    index=0
):
    '''
    CACTUS decoder function
    Parameters:
        compressed_path (str): path to the compressed represenation using CACTUS
        reconstructed_path (str): path where the pc should be reconstructed
        temp_folder (str): path to a folder where to save intermediate files
        codec_path (str): path to the codec
        codec (str): codec used, can be either draco or tmc13
        index (int): unique identifier for the pc (needed when parallelizing to avoid overwriting
            compressed represenations of other pcs)
    '''
    total_time = 0
    start_time = time()
    with open(compressed_path, "rb") as f:
        byte_classes = f.read(4)
        bitstring_classes = '{:032b}'.format(
            int.from_bytes(
                byte_classes,
                "little"
            )
        )
        covered_classes = [i for i in range(32) if bitstring_classes[i] == "1"]
        pcs = []
        classes = []
        total_time += time() - start_time

        for seg_class in covered_classes:
            start_time = time()
            pc_size = int.from_bytes(f.read(4), "little")
            pc = f.read(pc_size)

            comp_data_path = os.path.join(
                temp_folder,
                f"segment_{seg_class}_{index}.bin"
            )

            with open(comp_data_path, "wb") as temp_f:
                temp_f.write(pc)

            rec_data_path = os.path.join(
                temp_folder,
                f"temp_{seg_class}_{index}.ply"
            )
            total_time += time() - start_time

            total_time += decode(
                comp_data_path,
                rec_data_path,
                codec_path,
                codec,
                silence_output=True
            )

            temp_pc = read_ply_files(rec_data_path)
            if np.isnan(temp_pc).any() or np.isinf(temp_pc).any():
                print("Found a NaN or inf skipping")
                return None, False, total_time

            pcs.append(temp_pc)
            classes.append(seg_class * np.ones((temp_pc.shape[0], 1)))


        start_time = time()
        final_pc = np.vstack(pcs)
        total_time += time() - start_time
        write_ply_file(
            final_pc,
            reconstructed_path,
            attributes = np.vstack(classes),
            dtype = ["uint8"],
            names = ["class"],
            ascii_text=True
        )

        return final_pc.shape[0], True, total_time
