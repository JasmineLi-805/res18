# Generates the following helper files for ImgNet classification task:
#   - train_data.txt: the img_path and label for all training data
#   - val_data.txt: the img_path and label for all validation data
#   - test_data.txt: the img_path and label for all test data
#   - cls_mapping.txt: the mapping between cls_index, cls_id, cls_name

import os
import random
cwd = os.getcwd()

MAPPING_PATH = os.path.join(cwd, 'LOC_synset_mapping.txt')
MAPPING_SAVE_PATH = os.path.join(cwd, 'cls_mapping.txt')
TRAIN_DIR = os.path.join(cwd, 'ILSVRC/Data/CLS-LOC/train/')

def read_and_save_mapping(read_path, write_path):
    id2idx = {}
    output = []
    with open(read_path, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.strip()

            id = line[0:9].strip()
            name = line[10:]
            idx = len(id2idx)

            id2idx[id] = idx
            output.append(f'{idx}\t{id}\t{name}\n')
    
    with open(write_path, 'w') as f:
        f.writelines(output)
    
    return id2idx

def create_dataset_file(directory, id2idx, trpath, vpath, tepath):
    train_file = open(trpath, 'w')
    val_file = open(vpath, 'w')
    test_file = open(tepath, 'w')

    total = 0
    train_cnt = 0
    test_cnt = 0
    val_cnt = 0
    for id in id2idx:
        cls_dir = os.path.join(directory, id)
        assert os.path.isdir(cls_dir)
        for filename in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, filename)
            label = id2idx[id]
            out = f'{img_path}\t{label}\n'
            # partition the image into train/val/test set
            r = random.randint(1, 10)
            if r <= 7:
                train_file.write(out)
                train_cnt += 1
            elif r <= 9:
                val_file.write(out)
                val_cnt += 1
            else:
                test_file.write(out)
                test_cnt += 1
            total += 1
    print(f'total images={total}')
    print(f'train images={train_cnt}')
    print(f'val images={val_cnt}')
    print(f'test images={test_cnt}')
    train_file.close()
    val_file.close()
    test_file.close()

if __name__ == "__main__":
    print('start processing class id mapping')
    id2idx = read_and_save_mapping(MAPPING_PATH, MAPPING_SAVE_PATH)
    print('complete')

    print('creating data file')
    train_file_path = os.path.join(cwd, 'ILSVRC', 'train.txt')
    val_file_path = os.path.join(cwd, 'ILSVRC', 'val.txt')
    test_file_path = os.path.join(cwd, 'ILSVRC', 'test.txt')
    create_dataset_file(TRAIN_DIR, id2idx, train_file_path, val_file_path, test_file_path)

    print('ALL COMPLETED')

