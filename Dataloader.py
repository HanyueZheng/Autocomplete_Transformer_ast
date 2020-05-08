from VocabularyLoader import VocabularyLoader_char,VocabularyLoader_token, VocabularyLoader_token_ast
import random
import numpy as np
import torch
from torch.autograd import Variable
from HyperParameter import A_end_index,B_start_index

from functools import cmp_to_key as ctk


def foo(x, y):
    return len(x) - len(y)


def find_substring(string, substring):
    if len(string) == 0:
        return -1
    for i in range(len(string)):
        flag = 1
        for j in range(0, len(substring)):
            if string[i + j] != substring[j]:
                flag = 0
        if flag == 1:
            return i
    return -1


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
        self.content = "".join(lines).split()
        self.token_list = "".join(lines).replace('\n', ' ').replace('\t', ' ').split(' ')
        self.token_list = [i for i in self.token_list if (len(str(i))) != 0]
        self.file_len = len(self.token_list)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader_token(filename, self.device)

    def next_chunk(self):
        chunk = self.__random_chunk()
        input = chunk[:A_end_index]
        target = chunk[B_start_index:]
        print(input)
        print(target)
        return input, target

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + self.chunk_len
        if end_index > self.file_len:
            return self.vocabularyLoader.token_tensor(self.__random_chunk())
        else:
            print(self.token_list[start_index:end_index])
            return self.vocabularyLoader.token_tensor(self.token_list[start_index:end_index])


class DataLoader_token_kg():
    def __init__(self, filename, kg, chunk_len, device):
        self.kg = kg
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        self.content = "".join(lines).split()
        self.token_list = "".join(lines).replace('\n', ' ').replace('\t', ' ').split(' ')
        self.token_list = [i for i in self.token_list if (len(str(i))) != 0]
        self.file_len = len(self.token_list)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader_token(filename, self.device)

    def next_chunk(self):
        # TODO : add [UNK]
        chunk, content = self.__random_chunk()
        ents_list = []
        ents = [24] * self.chunk_len  # UNK = 24 to be modified
        contents = ""
        for i in range(len(content)):
            contents = contents + content[i] + " "
        for i in range(len(self.kg)):
            if contents.find(" " + self.kg[i] + " ") != -1:
                ents_list.append(self.kg[i])

        if len(ents_list) > 1:
            ents_list = sorted(ents_list, key=ctk(foo))

        for i in range(len(ents_list)):
            key = ents_list[i].strip().split()
            last_index = 0
            _ = content
            while True:
                try:
                    new_index = find_substring(_, key)
                    if new_index == -1:
                        break
                    last_index += new_index
                    for i in range(len(key)):
                        ents[last_index+i] = self.kg.index(" ".join(key))
                    last_index += len(key)
                    _ = content[last_index:]
                except Exception as e:
                    break
        ents = torch.Tensor(ents).long()
        input = chunk[:A_end_index]
        target = chunk[B_start_index:]
        ents = ents[:A_end_index]
        input = input.to(self.device)
        target = target.to(self.device)
        ent = ents.to(self.device)
        return input, ent, target

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + self.chunk_len
        if end_index > self.file_len:
            return self.vocabularyLoader.token_tensor(self.__random_chunk())
        else:
            # print(self.token_list[start_index:end_index])
            return self.vocabularyLoader.token_tensor(self.token_list[start_index:end_index]), \
                   self.token_list[start_index:end_index]



class DataLoader_token_ast():
        def __init__(self, filename, kg, ast_file, chunk_len, device, astdim):
            self.kg = kg
            self.astfile = ast_file
            with open(filename, 'r', encoding='UTF-8') as f:
                lines = f.readlines()
            self.content = "".join(lines)
            self.token_list = self.content.replace('\n', ' ').replace('\t', ' ').split(' ')
            self.token_list = [i for i in self.token_list if (len(str(i))) != 0]
            self.file_len = len(self.token_list)
            self.chunk_len = chunk_len
            self.device = device
            self.astdim = astdim
            self.vocabularyLoader = VocabularyLoader_token_ast(filename, ast_file, self.astdim, self.device)
            self.model = self.vocabularyLoader.word2vec()

        def find_path(self, filename):
            pathdic = {}
            key = ""
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if (line != ""):
                        if line[0] == "[":
                            pathdic[line] = []
                            key = line
                        else:
                            pathdic[key].append(line)
                            # line = f.readline()
                            # print(line)
                            # print(pathdic)
            # print(pathdic)
            return pathdic

        def path_encode(self, pathdic):
            path_encodedic = {}
            for k in pathdic.keys():
                path_encodedic[k] = []
                path = pathdic[k]
                for p in path:
                    pencode = self.vocabularyLoader.ast_path_tensor(p, self.model)
                    path_encodedic[k].append(pencode)
            return path_encodedic

        def next_chunk(self):
            # TODO : add [UNK]
            chunk, content = self.__random_chunk()
            ents_list = []
            ents = [24] * self.chunk_len  # UNK = 24 to be modified
            path_encodedic = self.path_encode(self.find_path(self.astfile))
            contents = ""

            for i in range(len(content)):
                contents = contents + content[i] + " "
            for i in range(len(self.kg)):
                if contents.find(" " + self.kg[i] + " ") != -1:
                    ents_list.append(self.kg[i])

            if len(ents_list) > 1:
                ents_list = sorted(ents_list, key=ctk(foo))

            for i in range(len(ents_list)):
                key = ents_list[i].strip().split()
                last_index = 0
                _ = content
                while True:
                    try:
                        new_index = find_substring(_, key)
                        if new_index == -1:
                            break
                        last_index += new_index
                        for i in range(len(key)):
                            ents[last_index + i] = self.kg.index(" ".join(key))
                        last_index += len(key)
                        _ = content[last_index:]
                    except Exception as e:
                        break
            ents = torch.Tensor(ents).long()

            path_addlist = torch.tensor([0] * (self.chunk_len-1)).float()
            keylist = []
            start = 0
            while contents.find("[", start) > 0:
                # print(path_encodedic)
                start = contents.find("[ iTC", start)
                # print("start: " + str(start))
                if contents.find("]", start) > 0:
                    end = contents.find("]", start)
                    # print("end:" + str(end))
                    key = contents[start: end + 1]
                    key = key.replace(" ", "") + "\n"
                    if key != "" and key in path_encodedic.keys():
                        keylist.append(key)
                    start = end
                else:
                    start = len(contents) - 1
            for k in keylist:
                ast_list = path_encodedic[k]

                for t in ast_list:
                    # t.resize(t.size(), path_tensor.size())
                    # print(t.size())
                    #                trans = nn.Embedding(t.size(), self.ast_token_num - 1)
                    path_addlist = path_addlist + torch.tensor(t).float()

            ast = torch.tensor(path_addlist).long()

            input_target_pair = []
            input = chunk[:-1]
            target = chunk[1:]
            ents = ents[:-1]
            ast = ast.to(self.device)
            input = input.to(self.device)
            target = target.to(self.device)
            ent = ents.to(self.device)

            #input_target_pair.append((input, ent, target, ast))
            return input, target, ent, ast



        def __random_chunk(self):
            start_index = random.randint(0, self.file_len - self.chunk_len)
            end_index = start_index + self.chunk_len
            if end_index > self.file_len:
                return self.vocabularyLoader.token_tensor(self.__random_chunk())
            else:
                return self.vocabularyLoader.token_tensor(self.token_list[start_index:end_index]), \
                       self.token_list[start_index:end_index]