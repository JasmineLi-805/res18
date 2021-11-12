
import os
import sys

VERBOSE = True

import numpy as np
import tqdm
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

from dataset.ImgNetDataset import ImgNetTextLineDataset

class ModelLoader:
    def __init__(self, model):
        self.model = model

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.process = None # data preprocessing

    def predict(self, dataloader):
        if VERBOSE:
            print(f'starting prediction...')
        images = []
        predictions = []
        labels = []
        with torch.no_grad():
            for batch_data in tqdm.tqdm(dataloader):
                if len(batch_data['label']) == 0:
                    print(f'batch_data[label] has length 0')
                    continue
                output = self.model(batch_data['img'].to(self.device))
                output = torch.nn.functional.softmax(output[0], dim=0)
                output = output.cpu().numpy()
                images.extend(batch_data['img'])
                predictions.extend([torch.argmax(txt) for txt in output])
                labels.extend(batch_data['label'])
        return images, predictions, labels

def build_eval_dataloader(file_path, eval_config):
    dataset = ImgNetTextLineDataset(file_path)
    assert len(dataset) > 0
    eval_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    return eval_loader

def get_stats(pred, data, output_dir=None):
    if VERBOSE:
        print(f'start calculating accuracy...')
    stats = {}
    idx_to_error = {}
    images = data['images']
    targ = data['labels']
    print(pred[:10])
    assert(len(pred) == len(targ))
    correct_cnt = 0
    for i in range(0, len(pred)):
        if pred[i] == targ[i]:
            correct_cnt += 1
        else:
            img  = images[i].permute(1, 2, 0).numpy()
            idx_to_error[i] = img
            if output_dir:
                img_name = output_dir + '/' + str(i) + '-' + str(pred[i]) + '.jpg'
                plt.imshow(img)
                plt.savefig(img_name)

    print(f'get stats: correct cnt={correct_cnt}, total cnt={len(pred)}, accuracy={1.0 * correct_cnt / len(pred)}')
    stats['accuracy'] = 1.0 * correct_cnt / len(pred)
    stats['idx_to_error'] = idx_to_error
    return stats

def init_args():
    import argparse
    if VERBOSE:
        print(f'parsing arguments')
    parser = argparse.ArgumentParser(description='PytorchOCR infer')
    parser.add_argument('--model_path', required=True, type=str, help='rec model path')
    # parser.add_argument('--img_path', required=True, type=str, help='img path for predict')
    parser.add_argument('--data_info', required=True, type=str, help='path to the text file containing img path and annotation')
    parser.add_argument('--output_dir', required=False, type=str, help='directory to store incorrectly labelled imgs', default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # args = init_args()
    if VERBOSE:
        print(f'loading model...')
    model = timm.create_model('resnet18', pretrained=True)
    model.eval()
    model = ModelLoader(model)

    print(f'building dataloader')
    dataset_file = os.path.join(os.getcwd(), 'ILSVRC', 'val.txt')
    eval_config = {
        'batch_size': 1,
        'shuffle': False
    }
    loader = build_eval_dataloader(dataset_file, eval)

    images, out, labels = model.predict(loader)
    out = [p[0][0] for p in out]
    stats = get_stats(out, {'labels': labels, 'images': images}, output_dir=None)
    accuracy = stats['accuracy']
    print(f'\ntotal # of images to predict: {len(labels)}; accuracy: {accuracy}')
