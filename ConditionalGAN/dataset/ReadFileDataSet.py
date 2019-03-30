from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class ReadFileDataSet(Dataset):
    def __init__(self,  file_path):
        self.file_path = file_path
        self.concentrations=[]
        self.target_color_label=[]
        with open(file_path, 'r') as f:
            for line in f.readlines():
                target_color = [float(x) for x in line.split(",")[:31]]
                concentrations = [float(x) for x in line.split(",")[-6:]]

                target_color_tensor = torch.Tensor(target_color) #31
                concentrations_tensor = torch.Tensor(concentrations) #2

                self.target_color_label.append(target_color_tensor)
                self.concentrations.append(concentrations_tensor)
        print('init dataset finished')

    def __getitem__(self, index):
        concentration=self.concentrations[index]
        target_color=self.target_color_label[index]
        return target_color, concentration

    def __len__(self):
        return len(self.concentrations)
