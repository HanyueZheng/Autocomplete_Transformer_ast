import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from Model import subsequent_mask
import matplotlib.pyplot as plt
import seaborn
from HyperParameter import epoches_of_loss_record,epoches_of_model_save

# =============================================================================
#
# Training : 模型训练
#
# =============================================================================

# 接下来，我们创建一个通用的训练和评分功能来跟踪损失。我们传递一个通用的损失计算函数，该函数还处理参数更新。

def run_epoch(stage,data_iter, model, loss_compute, nbatches,epoch=0):
	"Standard Training and Logging Function"
	start = time.time()
	total_tokens = 0
	total_loss = 0
	tokens = 0
	for i, batch in enumerate(data_iter):
	    out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
	    loss = loss_compute(out, batch.trg_y, batch.ntokens)

	    total_loss += loss.detach().cpu().numpy()
	    total_tokens += batch.ntokens.cpu().numpy()
	    tokens += batch.ntokens.cpu().numpy()
	    if i==nbatches-1:
		    elapsed = time.time() - start
		    print("%s  Loss: %f Tokens per Sec: %f" % (stage, loss.detach().cpu().numpy() / batch.ntokens.cpu().numpy(), tokens / elapsed))
		    if epoch%epoches_of_loss_record==0:
			    f = open("procedure.txt", "a+")
			    f.write("%s  Loss: %f Tokens per Sec: %f \n" % (stage, loss.detach().cpu().numpy() / batch.ntokens.cpu().numpy(), tokens / elapsed))
			    f.close()
		    start = time.time()
		    tokens = 0
	if epoch%epoches_of_model_save==0:
		torch.save(model, "transformer" + "{}".format(epoch) + ".model")


# 损失计算
class SimpleLossCompute:
	"一个简单的损失计算和训练函数"

	def __init__(self, generator, criterion, opt=None):
		self.generator = generator
		self.criterion = criterion
		self.opt = opt

	def __call__(self, x, y, norm):
		x = self.generator(x)
		loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
		loss.backward()
		if self.opt is not None:
			self.opt.step()
			self.opt.optimizer.zero_grad()
		return loss.data.item() * norm.float()

# 贪婪解码
# 为了简单起见，此代码使用贪婪解码来预测。
def greedy_decode(model, src, src_mask, max_len):
	memory = model.encode(src, src_mask)
	ys = torch.ones(1, 1).fill_(src[0][-1].cpu().numpy().item()).type_as(src.data)
	for i in range(max_len - 1):
		out = model.decode(memory, src_mask,
		                   Variable(ys),
		                   Variable(subsequent_mask(ys.size(1))
		                            .type_as(src.data)))
		prob = model.generator(out[:, -1])
		_, next_word = torch.max(prob, dim=1)
		next_word = next_word.data[0]
		ys = torch.cat([ys,
		                torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
	return ys