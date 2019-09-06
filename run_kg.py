import torch
import sys
import seaborn
import json
from torch.autograd import Variable
from Batch import Batch, Batch_kg
from Optim import NoamOpt, LabelSmoothing
from Model import make_model, make_model_kg
from Train import run_epoch, greedy_decode, beam_search_decode, SimpleLossCompute, run_epoch_kg
from Dataloader import DataLoader_char, DataLoader_token, DataLoader_token_kg
from HyperParameter import chunk_len, batch, nbatches, transformer_size, epoch_number, epoches_of_loss_record, \
	predict_length
# import numpy as np
# from torch import nn
# import torchsnooper
# from torchtext import data, datasets
# import spacy

seaborn.set_context(context="talk")

#cre文本匹配
#输入数据处理
def data_gen_char(dataloader, batch, nbatches):
    "为src-tgt复制任务生成随机数据"
    for i in range(nbatches):
        data_src = dataloader.next_chunk()[0].unsqueeze(0)
        data_tgt = dataloader.next_chunk()[1].unsqueeze(0)
        for j in range(batch - 1):
            data_src = torch.cat([data_src, dataloader.next_chunk()[0].unsqueeze(0)], 0)
            data_tgt = torch.cat([data_tgt, dataloader.next_chunk()[1].unsqueeze(0)], 0)

        src = Variable(data_src, requires_grad=False)
        tgt = Variable(data_tgt, requires_grad=False)
        yield Batch(src, tgt, 0)


# 共有nbatches*batch*(chunklen-1)条数据
def data_gen_token(dataloader, batch, nbatches, chunk_len, device):
    "为src-tgt复制任务生成随机数据"
    for i in range(nbatches):
        data_src = torch.empty(1, chunk_len - 1).long().to(device)
        data_tgt = torch.empty(1, 2).long().to(device)
        for k in range(batch):
            src_tgt_pair = dataloader.next_chunk()
            for j in range(0, len(src_tgt_pair)):
                data_src = torch.cat([data_src, src_tgt_pair[j][0].unsqueeze(0)])
                data_tgt = torch.cat([data_tgt, src_tgt_pair[j][1].unsqueeze(0)])
            data_src = data_src[1:]
            data_tgt = data_tgt[1:]
        src = Variable(data_src, requires_grad=False)
        tgt = Variable(data_tgt, requires_grad=False)
        yield Batch(src, tgt, -1)


# 共有nbatches*batch*(chunklen-1)条数据
def data_gen_token_kg(dataloader, batch, nbatches, chunk_len, device):
    "为src-tgt复制任务生成随机数据"
    for i in range(nbatches):
        data_src = torch.empty(1, chunk_len - 1).long().to(device)
        data_ent = torch.empty(1, chunk_len - 1).long().to(device)
        data_tgt = torch.empty(1, 2).long().to(device)
        for k in range(batch):
            src_tgt_pair = dataloader.next_chunk()
            for j in range(0, len(src_tgt_pair)):
                data_src = torch.cat([data_src, src_tgt_pair[j][0].unsqueeze(0)])
                data_ent = torch.cat([data_ent, src_tgt_pair[j][1].unsqueeze(0)])
                data_tgt = torch.cat([data_tgt, src_tgt_pair[j][2].unsqueeze(0)])
            data_src = data_src[1:]
            data_ent = data_ent[1:]
            data_tgt = data_tgt[1:]
        src = Variable(data_src, requires_grad=False)
        ent = Variable(data_ent, requires_grad=False)
        tgt = Variable(data_tgt, requires_grad=False)
        yield Batch_kg(src, ent, tgt, -1)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sys.argv.append('train')
    sys.argv.append('EN-ATP-V226.txt')
    sys.argv.append('token')
    sys.argv.append('transformer3000.model')
    sys.argv.append('The ATP software is the core')

    if (len(sys.argv) < 2):
        print("usage: lstm [train file | inference (words vocabfile) ]")
        print("e.g. 1: lstm train cre-utf8.txt")
        print("e.g. 2: lstm inference cre-utf8.txt words")
        sys.exit(0)

    method = sys.argv[1]
    if (method == "train"):
        filename = sys.argv[2]
        is_char_level = sys.argv[3] == 'char'

        if is_char_level:
            # TODO
            dataloader = DataLoader_char(filename, chunk_len, device)
            V = dataloader.vocabularyLoader.n_chars  # vocabolary size

            # kg_embed
            # with open("kg_embed/embedding.vec.json", "r", encoding='utf-8') as f:
            #     lines = json.loads(f.read())
            #     vecs = list()
            #     # vecs.append([0] * 100)  # CLS
            #     for (i, line) in enumerate(lines):
            #         if line == "ent_embeddings":
            #             for vec in lines[line]:
            #                 vec = [float(x) for x in vec]
            #                 vecs.append(vec)
            # embed = torch.FloatTensor(vecs)
            # embed = torch.nn.Embedding.from_pretrained(embed)
            # print(embed)  # Embedding(464, 100)

            criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
            criterion.cuda()
            model = make_model(V, V, N=transformer_size)
            model.cuda()
            model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

            for epoch in range(epoch_number):
                if epoch % epoches_of_loss_record == 0:
                    f = open("procedure.txt", "a+")
                    f.write("step:%d \n" % epoch)
                    f.close()
                print("step: ", epoch)
                model.train()
                run_epoch("train", data_gen_char(dataloader, batch, nbatches), model,
                          SimpleLossCompute(model.generator, criterion, model_opt), nbatches, epoch)
                model.eval()
                run_epoch("test ", data_gen_char(dataloader, batch, 1), model,
                          SimpleLossCompute(model.generator, criterion, None), nbatches, epoch)

        else:
            ents = []
            with open("kg_embed/entity2id.txt") as fin:
                fin.readline()
                for line in fin:
                    name, id = line.strip().split("\t")
                    ents.append(name)

            dataloader = DataLoader_token_kg(filename, ents, chunk_len, device)
            V = dataloader.vocabularyLoader.n_tokens  # vocabolary size

            criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
            criterion.cuda()
            model = make_model_kg(V, V, "kg_embed/embedding.vec.json", N=transformer_size)
            model.cuda()
            model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

            for epoch in range(epoch_number):
                if epoch % epoches_of_loss_record == 0:
                    f = open("procedure.txt", "a+")
                    f.write("step:%d \n" % epoch)
                    f.close()
                print("step: ", epoch)
                model.train()
                run_epoch_kg("train", data_gen_token_kg(dataloader, batch, nbatches, chunk_len, device), model,
                          SimpleLossCompute(model.generator, criterion, model_opt), nbatches, epoch)
                model.eval()
                run_epoch_kg("test ", data_gen_token_kg(dataloader, batch, nbatches, chunk_len, device), model,
                          SimpleLossCompute(model.generator, criterion, None), nbatches, epoch)



# if True:
#     spacy_en = spacy.load('en_core_web_sm')
#     spacy_de = spacy.load('de_core_news_sm')
#     print("good")
#
#     def tokenize_de(text):
#         return [tok.text for tok in spacy_de.tokenizer(text)]
#
#     def tokenize_en(text):
#         return [tok.text for tok in spacy_en.tokenizer(text)]
#
#     BOS_WORD = '<s>'
#     EOS_WORD = '</s>'
#     BLANK_WORD = '<blank>'
#     SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
#     TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD,
#                      eos_token = EOS_WORD, pad_token=BLANK_WORD)
#
#     MAX_LEN = 100
#     train, val, test = datasets.IWSLT.splits(
#         exts=('.de', '.en'), fields=(SRC, TGT),
#         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
#             len(vars(x)['trg']) <= MAX_LEN)
#     MIN_FREQ = 2
#     SRC.build_vocab(train.src, min_freq=MIN_FREQ)
#     TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
#
#
# class MyIterator(data.Iterator):
#     def create_batches(self):
#         if self.train:
#             def pool(d, random_shuffler):
#                 for p in data.batch(d, self.batch_size * 100):
#                     p_batch = data.batch(
#                         sorted(p, key=self.sort_key),
#                         self.batch_size, self.batch_size_fn)
#                     for b in random_shuffler(list(p_batch)):
#                         yield b
#
#             self.batches = pool(self.data(), self.random_shuffler)
#
#         else:
#             self.batches = []
#             for b in data.batch(self.data(), self.batch_size,
#                                 self.batch_size_fn):
#                 self.batches.append(sorted(b, key=self.sort_key))
#
#
# def rebatch(pad_idx, batch):
#     "Fix order in torchtext to match ours"
#     src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
#     return Batch(src, trg, pad_idx)
#
# if True:
#     pad_idx = TGT.vocab.stoi["<blank>"]
#     model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
#     model.cuda()
#     criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
#     criterion.cuda()
#     BATCH_SIZE = 12000
#     train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
#                             repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
#                             batch_size_fn=batch_size_fn, train=True)
#     valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
#                             repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
#                             batch_size_fn=batch_size_fn, train=False)
#     model_par = nn.DataParallel(model, device_ids=devices)
#
# if True:
#     model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
#             torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#     for epoch in range(10):
#         model_par.train()
#         run_epoch((rebatch(pad_idx, b) for b in train_iter),
#                   model_par,
#                   MultiGPULossCompute(model.generator, criterion,
#                                       devices=devices, opt=model_opt))
#         model_par.eval()
#         loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
#                           model_par,
#                           MultiGPULossCompute(model.generator, criterion,
#                           devices=devices, opt=None))
#         print(loss)
# else:
#     model = torch.load("iwslt.pt")