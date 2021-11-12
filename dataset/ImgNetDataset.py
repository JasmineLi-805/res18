import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImgNetTextLineDataset(Dataset):
    def __init__(self, dataset_file):
        """
        Used to process the dataset file with the format: `img_path\tlabel`

        :return None
        """
        self.labels = []
        with open(dataset_file, 'r', encoding='utf-8') as f_reader:
            for m_line in f_reader.readlines():
                params = m_line.split('\t')
                if len(params) == 2:
                    m_image_name, m_gt_text = params
                    m_gt_text = m_gt_text[:-1]
                    self.labels.append((m_image_name, m_gt_text))

    def _find_max_length(self):
        return max({len(_[1]) for _ in self.labels})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # get img_path and trans
        img_path, trans = self.labels[index]
        # read img
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dim = (256, 256)
        img = cv2.resize(img, dim)
        img = np.moveaxis(img, 0, -1)
        return {'img': img, 'label': trans}
