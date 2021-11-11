# Generates the following helper files for ImgNet classification task:
#   - train_data.txt: the img_path and label for all training data
#   - val_data.txt: the img_path and label for all validation data
#   - test_data.txt: the img_path and label for all test data
#   - cls_mapping.txt: the mapping between cls_index, cls_id, cls_name

import os
cwd = os.getcwd()

MAPPING_PATH = os.path.join(cwd, 'LOC_synset_mapping.txt')
MAPPING_SAVE_PATH = os.path.join(cwd, 'cls_mapping.txt')
TRAIN_DIR = os.path.join(cwd, 'ILSVRC/Data/CLS-LOC/train/')
VAL_DIR = os.path.join(cwd, 'ILSVRC/Data/CLS-LOC/val/')
TEST_DIR = os.path.join(cwd, 'ILSVRC/Data/CLS-LOC/test/')

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

def create_dataset_file(directory, id2idx, save_path):
    output_file = open(save_path, 'w')
    for id in id2idx:
        cls_dir = os.path.join(directory, id)
        assert os.path.isdir(cls_dir)
        for filename in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, filename)
            label = id2idx[id]
            out = f'{img_path}\t{label}\n'
            output_file.write(out)
    output_file.close()

if __name__ == "__main__":
    print('start processing class id mapping')
    id2idx = read_and_save_mapping(MAPPING_PATH, MAPPING_SAVE_PATH)
    print('complete')

    print('creating training data file')
    train_file_path = os.path.join(cwd, 'ILSVRC', 'train.txt')
    create_dataset_file(TRAIN_DIR, id2idx, train_file_path)

    print('creating validation data file')
    val_file_path = os.path.join(cwd, 'ILSVRC', 'val.txt')
    create_dataset_file(VAL_DIR, id2idx, val_file_path)

    print('creating test data file')
    test_file_path = os.path.join(cwd, 'ILSVRC', 'test.txt')
    create_dataset_file(TEST_DIR, id2idx, test_file_path)

    print('ALL COMPLETED')

