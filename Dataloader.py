from VocabularyLoader import VocabularyLoader_char,VocabularyLoader_token
import random
import torchsnooper
import numpy as np 
import torch
from torch.autograd import Variable
import os


class DataLoader_char():
    def __init__(self, filename, chunk_len, device):
        with open(filename,'r',encoding='UTF-8') as f:
            lines=f.readlines()
        self.content = "".join(lines)
        self.file_len = len(self.content)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader_char(filename, self.device)

    def next_chunk(self):
        chunk = self.__random_chunk()
        input = chunk[:-1]
        target = chunk[1:]
        return input, target

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + self.chunk_len
        if end_index > self.file_len:
            return self.vocabularyLoader.char_tensor(self.__random_chunk())
        else:
            return self.vocabularyLoader.char_tensor(self.content[start_index:end_index])


class DataLoader_token():
    def __init__(self, filename, chunk_len, device):
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        self.content = "".join(lines)
        self.token_list = self.content.replace('\n', ' ').replace('\t', ' ').split(' ')
        self.token_list = [i for i in self.token_list if (len(str(i))) != 0]
        self.file_len = len(self.token_list)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader_token(filename, self.device)

    def next_chunk(self):
        chunk = self.__random_chunk()
        input_target_pair = []
        for i in range(1, self.chunk_len):
            input = torch.zeros(self.chunk_len -1 ).long()
            for j in range(self.chunk_len-i-1):
                input[j] = self.vocabularyLoader.n_tokens-1
            input[-i:] = chunk[:i]
            target = chunk[i-1:i+1]
            input = input.to(self.device)
            target = target.to(self.device)
            input_target_pair.append((input, target))
        return input_target_pair

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + self.chunk_len
        if end_index > self.file_len:
            return self.vocabularyLoader.token_tensor(self.__random_chunk())
        else:
            return self.vocabularyLoader.token_tensor(self.token_list[start_index:end_index])


class DataLoader_token_kg():
    def __init__(self, filename, kg, chunk_len, device):
        self.kg = kg
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        self.content = "".join(lines)
        self.token_list = self.content.replace('\n', ' ').replace('\t', ' ').split(' ')
        self.token_list = [i for i in self.token_list if (len(str(i))) != 0]
        self.file_len = len(self.token_list)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader_token(filename, self.device)

    def next_chunk(self):
        # TODO : add [UNK]
        chunk, content = self.__random_chunk()
        ents_list = []
        ents = [" "] * self.chunk_len
        contents = ""
        for i in range(len(content)):
            contents = contents + content[i] + " "
        # print(contents)
        for i in range(len(self.kg)):
            if contents.find(" " + self.kg[i] + " ") != -1:
                ents_list.append(self.kg[i])
        # print(ents_list)
        for i in range(len(ents_list)):
            key = ents_list[i].strip().split()[0]
            if content.index(key) >= 0:
                ents[content.index(key)] = key
        # TODO revise the ents

        ents = self.vocabularyLoader.token_tensor(ents)
        input_target_pair = []
        for i in range(1, self.chunk_len):
            input = torch.zeros(self.chunk_len - 1).long()
            ent = torch.zeros(self.chunk_len).long()
            for j in range(self.chunk_len - i - 1):
                input[j] = self.vocabularyLoader.n_tokens - 1
            input[-i:] = chunk[:i]
            target = chunk[i - 1:i + 1]
            ent[-i:] = ents[:i]
            input = input.to(self.device)
            target = target.to(self.device)
            ent = ent.to(self.device)
            input_target_pair.append((input, target, ent))
        return input_target_pair

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + self.chunk_len
        if end_index > self.file_len:
            return self.vocabularyLoader.token_tensor(self.__random_chunk())
        else:
            return self.vocabularyLoader.token_tensor(self.token_list[start_index:end_index]), \
                   self.token_list[start_index:end_index]