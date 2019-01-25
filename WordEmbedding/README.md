# Word Embedding

# 0.Introduction

## 1. What is Word embedding?

Word embedding, is the collective name for as set of language modeling and feature learning techiques in natural language processing(NLP) where words or phrases form the vocabulary are mapped to vectors of real numbers.

Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension.

# 1.About

## 1. How to understand Word embedding?

Here is an answer i posted in Zhihu before:[有谁可以解释下word embedding? - lightsmile的回答 - 知乎](https://www.zhihu.com/question/32275069/answer/563578516)

# 2.Kinds

There are kinds of Word embedding trainning methods, such as Word2vec,Glove,Fasttext,Bert, etc.

# 3.Use in code

The pre-trained vector files are just plain text format. Commonly, each line contains a word and its vector, each value is separated by space.


Glove and Word2vec are the most popular models in training word embeddings, both the outputs are text format,the little difference is that Word2vec starts with one line contains the nums of the training vocab and the dimension of each vector.

Here are the examples:

The Glove Word embedding's format like this:

```bash
word1 0.123 0.134 0.532 0.152
word2 0.934 0.412 0.532 0.159
word3 0.334 0.241 0.324 0.18
...
word9 0.334 0.241 0.324 0.188
```

the Word2vec Word embedding's format like this:

```bash
9 4   # 这一行包含向量的数量及其维度
word1 0.123 0.134 0.532 0.152
word2 0.934 0.412 0.532 0.159
word3 0.334 0.241 0.324 0.188
...
word9 0.334 0.241 0.324 0.188
```


# 4.Some resources

- [fastText Pre-trained word vectors](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
- [100+ Chinese Word Vectors 上百种预训练中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)

# 5.References

- [如何gensim加载glove训练的词向量](https://www.jianshu.com/p/c2a9d3e76706)
- [Word embedding](https://en.wikipedia.org/wiki/Word_embedding)





