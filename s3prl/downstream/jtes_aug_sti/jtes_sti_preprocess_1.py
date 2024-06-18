#!/usr/bin/env python3

# JTES STI Fold 1
import sys
import glob
import os
import json


def main(data_dir):
    file1 = glob.glob(os.path.join(data_dir + 'wav/*/*/', '*.wav'))
    file2 = glob.glob(os.path.join(data_dir + 'arhmm/*/*/', '*.wav'))
    file3 = glob.glob(os.path.join(data_dir + 'aug_ir/*/*/', '*.wav'))
    file4 = glob.glob(os.path.join(data_dir + 'aug_noise/*/*/', '*.wav'))
    files = file1 + file2 + file3 + file4
    files.sort()

    data_train = []
    data_test = []
    data_val = []

    for file in files:
        # processing file
        print("Processing... ", file)
        lab_str = os.path.basename(os.path.dirname(file))
        # use speaker 1-45 and text 1-40 as training, the rest as test
        if int(os.path.basename(file)[1:3]) in range(1, 46):
            if int(os.path.basename(file)[8:10]) in range(1, 41):
                data_train.append({
                    "path": file,
                    "label": lab_str,
                    "speaker": int(os.path.basename(file)[1:3]),
                })
        elif int(os.path.basename(file)[1:3]) in range(46, 51):
            if os.path.basename(file)[10:14] != '.wav':  # Filter augmenteds
                continue
            if int(os.path.basename(file)[8:10]) in range(41, 51):
                data_test.append({
                    "path": file,
                    "label": lab_str,
                    "speaker": int(os.path.basename(file)[1:3]),
                })

    out_dir = os.path.join(data_dir, 'jtes_aug_sti', 'meta_data', 'sti1')
    os.makedirs(out_dir, exist_ok=True)
    data_train = {
        'labels': {'ang': 0, 'joy': 1, 'neu': 2, 'sad': 3},
        'meta_data': data_train
    }
    data_test = {
        'labels': {'ang': 0, 'joy': 1, 'neu': 2, 'sad': 3},
        'meta_data': data_test
    }
    with open(f"{out_dir}/train_meta_data.json", 'w') as f1:
        json.dump(data_train, f1)
    with open(f"{out_dir}/test_meta_data.json", 'w') as f2:
        json.dump(data_test, f2)

    print(f"meta data saved to {out_dir}")
    print(f"length of train: {len(data_train['meta_data'])}")
    print(f"length of test: {len(data_test['meta_data'])}")


if __name__ == '__main__':
    main(sys.argv[1])
