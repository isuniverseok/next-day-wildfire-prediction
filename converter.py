import os
from glob import glob
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

INPUT_FEATURES = [
    "elevation",
    "th",
    "vs",
    "tmmn",
    "tmmx",
    "sph",
    "pr",
    "pdsi",
    "NDVI",
    "population",
    "erc",
    "PrevFireMask",
]
OUTPUT_FEATURES = ["FireMask"]

DATA_STATS = {
    "elevation": (0.0, 3141.0, 657.3003, 649.0147),
    "pdsi": (-6.12974870967865, 7.876040384292651, -0.0052714925, 2.6823447),
    "NDVI": (-9821.0, 9996.0, 5157.625, 2466.6677),
    "pr": (0.0, 44.53038024902344, 1.7398051, 4.482833),
    "sph": (0.0, 1.0, 0.0071658953, 0.0042835088),
    "th": (0.0, 360.0, 190.32976, 72.59854),
    "tmmn": (253.15, 298.94891357421875, 281.08768, 8.982386),
    "tmmx": (253.15, 315.09228515625, 295.17383, 9.815496),
    "vs": (0.0, 10.024310074806237, 3.8500874, 1.4109988),
    "erc": (0.0, 106.24891662597656, 37.326267, 20.846027),
    "population": (0.0, 2534.06298828125, 25.531384, 154.72331),
    "PrevFireMask": (-1.0, 1.0, 0.0, 1.0),
    "FireMask": (-1.0, 1.0, 0.0, 1.0),
} #for reference -> check the paper 


def convert_tfrecord_to_npz(file_path, output_directory, npz_filename):
    dataset = tf.data.TFRecordDataset(file_path)
    feature_names = INPUT_FEATURES + OUTPUT_FEATURES
    sample_size = 64  
    features_dict = {
        name: tf.io.FixedLenFeature(shape=[sample_size, sample_size], dtype=tf.float32)
        for name in feature_names
    }

    def _parse_record(example_proto):
        features = tf.io.parse_single_example(example_proto, features_dict)
        return {key: features.get(key) for key in feature_names}

    out_features = defaultdict(list)  
    for i, record in tqdm(enumerate(dataset)):
        parsed_record = _parse_record(record)
        # Convert tensors to numpy
        for var_name, val in parsed_record.items():
            out_features[var_name].append(val.numpy())

    for var_name, val in out_features.items():
        out_features[var_name] = np.stack(val, axis=0)

    if not npz_filename.endswith(".npz"):
        npz_filename += ".npz"

    out_features["data_stats"] = DATA_STATS

    np.savez(os.path.join(output_directory, npz_filename), **out_features)


if __name__ == "__main__":

    data_path = "arneshdadhich/Downloads/next-day-wildfire.tfrecords" #replace with the respective path
    output_dir = f"{data_path}_npz" 
    train_files = sorted(glob(os.path.join(data_path, "*train*.tfrecord")))
    eval_files = sorted(glob(os.path.join(data_path, "*eval*.tfrecord")))
    test_files = sorted(glob(os.path.join(data_path, "*test*.tfrecord")))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    convert_tfrecord_to_npz(train_files, output_dir, npz_filename="train.npz")
    convert_tfrecord_to_npz(eval_files, output_dir, npz_filename="val.npz")
    convert_tfrecord_to_npz(test_files, output_dir, npz_filename="test.npz")

    