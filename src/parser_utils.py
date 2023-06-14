import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--codec",
        type=str,
        default="draco",
        help="codec used to compress the partitions of the PC, can be either draco or TMC13 [default: draco]"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="../config/config.json",
        help="json config file to tell how much to quantize each class"
    )

    parser.add_argument(
        "--codecs_path",
        type=str,
        default="../../",
        help="relative (w.r.t the src folder) or absolute "
        "path to the codecs folder which should be named as when cloning "
        "from github (e.g. in the case of TMC13 the root of the "
        "codec should be named mpeg-pcc-tmc13)"
    )

    FLAGS = parser.parse_args()

    return FLAGS
