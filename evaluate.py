
import os
import sys

VERBOSE = True

import numpy as np
import tqdm
import torch
from torch import nn

import matplotlib.pyplot as plt
import cv2

class ModelLoader:
    def __init__(self, model, batch_size=16):
        self.model = model

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.process = None # data preprocessing
        self.batch_size = batch_size

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
    eval_config['dataset']['file'] = file_path
    eval_config['dataset']['alphabet'] = os.getcwd() + '/ccpd-data/dictionary.txt'
    eval_loader = build_dataloader(eval_config)
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
    args = init_args()
    if VERBOSE:
        print(f'loading model...')
    model = RecInfer(args.model_path)
    print(f'building dataloader')
    loader = build_eval_dataloader(args.data_info, model.data_config)

    images, out, labels = model.predict(loader)
    out = [p[0][0] for p in out]
    stats = get_stats(out, {'labels': labels, 'images': images}, output_dir=args.output_dir)
    accuracy = stats['accuracy']
    print(f'\ntotal # of images to predict: {len(labels)}; accuracy: {accuracy}')


    # data_info = '/Users/jasmineli/Desktop/PytorchOCR/ccpd-data/val.txt'
    # rnn_model = '/Users/jasmineli/Desktop/output/latest.pth'
    # fc_model = '/Users/jasmineli/Desktop/output_noneck/best.pth'

    # print(f'Starting...')
    # print(f'loading model...')
    # crnn= RecInfer(rnn_model)
    # fconv = RecInfer(fc_model)
    # print(f'building dataloader')
    # # assert crnn.data_config == fconv.data_config
    # loader = build_eval_dataloader(data_info, crnn.data_config)

    # output_dir = None
    # crnn_images, crnn_out, crnn_labels = crnn.predict(loader)
    # crnn_out = [p[0][0] for p in crnn_out]
    # crnn_stats = get_stats(crnn_out, {'labels': crnn_labels, 'images': crnn_images}, output_dir=output_dir)
    # # accuracy = crnn_stats['accuracy']
    # # print(f'\ntotal # of images to predict: {len(labels)}; accuracy: {accuracy}')

    # fconv_images, fconv_out, fconv_labels = fconv.predict(loader)
    # fconv_out = [p[0][0] for p in fconv_out]
    # fconv_stats = get_stats(fconv_out, {'labels': fconv_labels, 'images': fconv_images}, output_dir=output_dir)

    # fconv_idx_error = fconv_stats['idx_to_error']
    # crnn_idx_error = crnn_stats['idx_to_error']
    # for idx in fconv_idx_error:
    #     if idx in crnn_idx_error:
    #         direc = '/Users/jasmineli/Desktop/alpr/error_fconv-and-rnn/'
    #         crnn_pred = crnn_out[idx]
    #         fconv_pred = fconv_out[idx]
    #         img_name = direc +str(idx) + '-' + fconv_pred + '-' + crnn_pred + '.jpg'
    #         plt.imshow(fconv_idx_error[idx])
    #         plt.savefig(img_name)
    #     else:
    #         direc = '/Users/jasmineli/Desktop/alpr/error_fconv-not-rnn/'
    #         fconv_pred = fconv_out[idx]
    #         img_name = direc + str(idx) + '-' + fconv_pred + '.jpg'
    #         plt.imshow(fconv_idx_error[idx])
    #         plt.savefig(img_name)

    # for idx in crnn_idx_error:
    #     if idx not in fconv_idx_error:
    #         direc = '/Users/jasmineli/Desktop/alpr/error_rnn-not-fconv/'
    #         crnn_pred = crnn_out[idx]
    #         img_name = direc + str(idx) + '-' + crnn_pred + '.jpg'
    #         plt.imshow(crnn_idx_error[idx])
    #         plt.savefig(img_name)