from VocabularyLoader import VocabularyLoader
import random
import torchsnooper
import numpy as np 
import torch
from torch.autograd import Variable
import os

class DataLoader():
    def __init__(self, filename, chunk_len, device):
        with open(filename,'r',encoding='UTF-8') as f:
            lines=f.readlines()
        self.content = "".join(lines)
        self.file_len = len(self.content)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader(filename, self.device)


    def next_chunk(self):
        chunk = self.__random_chunk()
        input = chunk[:-1]
        target = chunk[1:]
        return input, target

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + self.chunk_len
        if(end_index > self.file_len):
            return self.vocabularyLoader.char_tensor(self.__random_chunk())
        else:
            return self.vocabularyLoader.char_tensor(self.content[start_index:end_index])
