import os
import json
import random
import argparse
import numpy as np
import re


def name_ends_with_digit(filename: str) -> bool:
    # Use it optionally to filter out files 
    # return re.match(r'.*(00[6-9]|01[0-8])$', filename)
    # return (filename[-6:-4] != '12')
    return True


def get_list_filenames(root: str) -> list:
    list_filenames = []
    
    for path, _, files in os.walk(root):
        for name in files:
            if name_ends_with_digit(name):
                list_filenames.append(os.path.join(path, name))
                
    return list_filenames


def create_data_index(list_filenames: list) -> dict:
    
    l_keys = []
    l_values = []
    
    for name in list_filenames:
        key = os.path.basename(os.path.normpath(name))
        if key.endswith('.nc'):
            key = key[:-3]

        l_keys.append(key)
        l_values.append(name)

    return dict(zip(l_keys, l_values))


def main(gdps: str, hrdps: str, train_val_test_split: list, output: str):

    if np.sum(train_val_test_split) != 100:
        raise ValueError('Sum of train, val and test split must be 100')
    
    # create index for gdps and hrdps individually
    list_filenames = []
    for path in gdps:
        list_filenames.extend(get_list_filenames(path))
    random.shuffle(list_filenames)
    gdps_indexes = create_data_index(list_filenames)
    
    list_filenames = []
    for path in hrdps:
        list_filenames.extend(get_list_filenames(path))
    random.shuffle(list_filenames)
    hrdps_indexes = create_data_index(list_filenames)

    l_keys = []
    l_values = []
    i = 0

    # get pairs of gdps and hrdps
    for key, value_gdps in gdps_indexes.items():
        if key in hrdps_indexes: 
            l_keys.append(i)
            l_values.append([value_gdps, hrdps_indexes[key]])
            i += 1


    # train/val/test split
    dataIndex = dict(zip(l_keys, l_values))
    total_len = len(dataIndex)

    start_val = int(np.ceil(train_val_test_split[0]/100 * total_len))              
    start_test = int(start_val + np.ceil(train_val_test_split[1]/100 * total_len)) 
    
    dataIndexTrain = {k: v for k, v in list(dataIndex.items())[:start_val]}
    dataIndexVal = {k - start_val: v for k, v in list(dataIndex.items())[start_val:start_test]}
    dataIndexTest = {k - start_test: v for k, v in list(dataIndex.items())[start_test:]}

    with open(f'{output}Train.json', 'w') as fp:
        json.dump(dataIndexTrain, fp)

    with open(f'{output}Val.json', 'w') as fp:
        json.dump(dataIndexVal, fp)

    with open(f'{output}Test.json', 'w') as fp:
        json.dump(dataIndexTest, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate index')
    parser.add_argument('--gdps_paths', nargs='+', dest='gdps', required=True, type=str, help='Path to GDPS')
    parser.add_argument('--hrdps_paths', nargs='+', dest='hrdps', required=True, type=str, help='Path to HRDPS')
    parser.add_argument('--train_val_test_split', nargs='+', dest='train_val_test_split', type=int, help='Train/Val/Test split', default=[75, 12.5, 12.5])
    parser.add_argument('--output', dest='output', type=str, help='Output path', default='')

    args = vars(parser.parse_args())
    main(**args)