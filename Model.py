import torch.nn as nn
import numpy as np
import torch
import copy
from Sublayers import MultiHeadedAttention,PositionwiseFeedForward,Generator,LayerNorm
from Layers import EncoderLayer,DecoderLayer,clones
from Embed import Embeddings,PositionalEncoding

# =============================================================================
#
# Full Model : 整体模型
#
# =============================================================================
# 定义一个函数，它接受超参数并生成完整的模型。
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
	"从超参数构造模型"
	c = copy.deepcopy
	attn = MultiHeadedAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)
	model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn),c(ff), dropout), N),
		nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
		nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
		Generator(d_model, tgt_vocab))

	# 从代码来看，使用 Glorot / fan_avg初始化参数很重要。
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)
	return model

class EncoderDecoder(nn.Module):
	"""
	标准编码器-解码器结构，本案例及其他各模型的基础。
	"""

	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator

	def forward(self, src, tgt, src_mask, tgt_mask):
		"处理屏蔽的源序列与目标序列"
		return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

	def encode(self, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask)

	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# =============================================================================
#
# Encoder 编码器
#
# =============================================================================
class Encoder(nn.Module):
	"核心编码器是N层堆叠"

	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, mask):
		"依次将输入的数据（及屏蔽数据）通过每个层"
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)

# =============================================================================
#
# Decoder 解码器
#
# =============================================================================
# 解码器也由一个N=6个相同层的堆栈组成。
class Decoder(nn.Module):
	"带屏蔽的通用N层解码器"

	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, memory, src_mask, tgt_mask):
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)


def subsequent_mask(size):
	"屏蔽后续位置"
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0

# plt.figure(figsize=(5, 5))
# plt.imshow(subsequent_mask(20)[0])