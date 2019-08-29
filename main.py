import torch
import sys
import seaborn
from torch.autograd import Variable
from Batch import Batch
from Optim import NoamOpt,LabelSmoothing
from Model import make_model
from Train import run_epoch,greedy_decode,SimpleLossCompute
from Dataloader import DataLoader
from HyperParameter import chunk_len,batch,nbatches,transformer_size,epoch_number,epoches_of_loss_record,predict_length
import numpy as np
from torch import nn
import torchsnooper
from torchtext import data, datasets
import spacy

seaborn.set_context(context="talk")

#cre文本匹配
#输入数据处理
def data_gen(dataloader,batch,nbatches):
	"为src-tgt复制任务生成随机数据"
	for i in range(nbatches):
		data_src = dataloader.next_chunk()[0].unsqueeze(0)
		data_tgt = dataloader.next_chunk()[1].unsqueeze(0)
		for j in range(batch-1):
			data_src=torch.cat([data_src,dataloader.next_chunk()[0].unsqueeze(0)],0)
			data_tgt=torch.cat([data_tgt,dataloader.next_chunk()[1].unsqueeze(0)],0)

		src = Variable(data_src, requires_grad=False)
		tgt = Variable(data_tgt, requires_grad=False)
		yield Batch(src, tgt, 0)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sys.argv.append('train')
    sys.argv.append('cre-utf8.txt')
    sys.argv.append('transformer100.model')
    sys.argv.append('编号')

    if(len(sys.argv)<2):
        print("usage: lstm [train file | inference (words vocabfile) ]")
        print("e.g. 1: lstm train cre-utf8.txt")
        print("e.g. 2: lstm inference cre-utf8.txt words")
        sys.exit(0)

    method = sys.argv[1]
    if(method == "train"):
        filename = sys.argv[2]
        dataloader = DataLoader(filename, chunk_len, device)
        # print(dataloader.vocabularyLoader.n_chars)

        V = dataloader.vocabularyLoader.n_chars  #vocabolary size
        criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
        criterion.cuda()
        model = make_model(V, V, N=transformer_size)
        model.cuda()
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
		                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        for epoch in range(epoch_number):
            if epoch%epoches_of_loss_record==0:
                f = open("procedure.txt", "a+")
                f.write("step:%d \n" % epoch)
                f.close()
            print("step: ",epoch)
            model.train()
            run_epoch("train",data_gen(dataloader,batch,nbatches), model,
				SimpleLossCompute(model.generator, criterion, model_opt),nbatches,epoch)
            model.eval()
            run_epoch("test ",data_gen(dataloader,batch,1), model,
					SimpleLossCompute(model.generator, criterion, None),nbatches,epoch)

    elif (method == "inference"):
        filename = sys.argv[2]
        trained_model_name=sys.argv[3]
        words = sys.argv[4]
        model = torch.load(trained_model_name).cuda()
        model.eval()
        dataloader = DataLoader(filename, chunk_len, device)
        src = Variable(dataloader.vocabularyLoader.char_tensor(words).unsqueeze(0))
        src_mask = Variable((src != 0).unsqueeze(-2))
        output_embed=greedy_decode(model, src, src_mask, max_len=predict_length)[0].cpu().numpy()
        result=""
        for i in output_embed:
            result+=dataloader.vocabularyLoader.index2char[i]
        print(result[1:])






# if True:
# 	spacy_en = spacy.load('en_core_web_sm')
# 	spacy_de = spacy.load('de_core_news_sm')
# 	print("good")
#
# 	def tokenize_de(text):
# 		return [tok.text for tok in spacy_de.tokenizer(text)]
#
# 	def tokenize_en(text):
# 		return [tok.text for tok in spacy_en.tokenizer(text)]
#
# 	BOS_WORD = '<s>'
# 	EOS_WORD = '</s>'
# 	BLANK_WORD = '<blank>'
# 	SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
# 	TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD,
# 	                 eos_token = EOS_WORD, pad_token=BLANK_WORD)
#
# 	MAX_LEN = 100
# 	train, val, test = datasets.IWSLT.splits(
# 	    exts=('.de', '.en'), fields=(SRC, TGT),
# 	    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
# 	        len(vars(x)['trg']) <= MAX_LEN)
# 	MIN_FREQ = 2
# 	SRC.build_vocab(train.src, min_freq=MIN_FREQ)
# 	TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
#
#
# class MyIterator(data.Iterator):
# 	def create_batches(self):
# 		if self.train:
# 			def pool(d, random_shuffler):
# 				for p in data.batch(d, self.batch_size * 100):
# 					p_batch = data.batch(
# 						sorted(p, key=self.sort_key),
# 						self.batch_size, self.batch_size_fn)
# 					for b in random_shuffler(list(p_batch)):
# 						yield b
#
# 			self.batches = pool(self.data(), self.random_shuffler)
#
# 		else:
# 			self.batches = []
# 			for b in data.batch(self.data(), self.batch_size,
# 			                    self.batch_size_fn):
# 				self.batches.append(sorted(b, key=self.sort_key))
#
#
# def rebatch(pad_idx, batch):
# 	"Fix order in torchtext to match ours"
# 	src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
# 	return Batch(src, trg, pad_idx)
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
# 	model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
# 	        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
# 	for epoch in range(10):
# 	    model_par.train()
# 	    run_epoch((rebatch(pad_idx, b) for b in train_iter),
# 	              model_par,
# 	              MultiGPULossCompute(model.generator, criterion,
# 	                                  devices=devices, opt=model_opt))
# 	    model_par.eval()
# 	    loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
# 	                      model_par,
# 	                      MultiGPULossCompute(model.generator, criterion,
# 	                      devices=devices, opt=None))
# 	    print(loss)
# else:
#     model = torch.load("iwslt.pt")