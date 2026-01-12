# -*-coding: utf-8 -*-
# 先运行 word_seg进行中文分词，然后再进行word_similarity计算
# 将Word转换成Vec，然后计算相似度 
from gensim.models import word2vec
import multiprocessing


# 如果目录中有多个文件，可以使用PathLineSentences
segment_folder = './three_kingdoms/segment'
# 切分之后的句子合集
sentences = word2vec.PathLineSentences(segment_folder)
# 设置模型参数，进行训练
model = word2vec.Word2Vec(sentences, vector_size=100, window=3, min_count=1)

print(model.wv.most_similar(positive=['曹操']))
print(model.wv.most_similar(positive=['曹操', '刘备'], negative=['张飞']))
