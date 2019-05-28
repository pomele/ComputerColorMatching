from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class ReadFileDataSet(Dataset):
    def __init__(self,  file_path):
        self.file_path = file_path
        self.concentrations=[]
        self.target_color_label=[]
        self.cur_noise=[]
        with open(file_path, 'r') as f:
            for line in f.readlines():
                target_color = [float(x) for x in line.split(",")[:31]]
                concentrations = [float(x) for x in line.split(",")[-5:-1]]
                cur_noise = [float(x) for x in line.split(",")[-1:]]

                print('这是当前的noise：+++++++++++++++')
                print(cur_noise)
                target_color_tensor = torch.Tensor(target_color)
                concentrations_tensor = torch.Tensor(concentrations)
                cur_noise_tensor = torch.Tensor(cur_noise)

                self.target_color_label.append(target_color_tensor)
                self.concentrations.append(concentrations_tensor)
                self.cur_noise.append(cur_noise_tensor)
        print('init dataset finished')

    def __getitem__(self, index):
        concentration=self.concentrations[index]
        target_color=self.target_color_label[index]
        cur_noise=self.cur_noise[index]
        return target_color, concentration, cur_noise

    def __len__(self):
        return len(self.concentrations)
