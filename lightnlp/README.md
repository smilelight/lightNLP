# lightNLP, lightsmile个人的自然语言处理框架

## 前言

依据自然语言处理四大任务，框架主要设计为有以下功能：

- 序列标注， Sequence Labeling
- 文本分类， Text Classification
- 句子关系， Sentence Relation
- 文本生成， Text Generation

因此将有四个主要的功能模块：sl（序列标注）、tc（文本分类）、sr（句子关系）、tg（文本生成）和其他功能模块如we（字嵌入）。

## 安装

本项目基于Pytorch1.0

```bash
pip install lightNLP
```

## 模型

BiLstm-Crf

## 训练数据标签

BIO

训练数据示例如下：

```bash
清 B_Time
明 I_Time
是 O
人 B_Person
们 I_Person
祭 O
扫 O
先 B_Person
人 I_Person
， O
怀 O
念 O
追 O
思 O
的 O
日 B_Time
子 I_Time
。 O

正 O
如 O
宋 B_Time
代 I_Time
诗 B_Person
人 I_Person
```

## 使用

当前只有sl下属的ner功能。

### 训练

```python
from lightnlp.sl import NER

# 创建NER对象
ner_model = NER()

train_path = '/home/lightsmile/Download/ner/test.sample.txt'
dev_path = '/home/lightsmile/Download/ner/train.sample.txt'
vec_path = '/home/lightsmile/Projects/NLP/ChineseEmbedding/model/token_vec_300.bin'

# 只需指定训练数据路径，预训练字向量可选，开发集路径可选，模型保存路径可选。
ner_model.train(train_path, vectors_path=vec_path, dev_path=dev_path)
```

### 测试

```python
# 加载模型，默认当前目录下的`saves`目录
ner_model.load()
# 对train_path下的测试集进行读取测试
ner_model.test(train_path)
```

### 预测

```python
# 加载模型，默认当前目录下的`saves`目录
ner_model.load()

from pprint import pprint

pprint(ner_model.predict('另一个很酷的事情是，通过框架我们可以停止并在稍后恢复训练。'))
```

预测结果：

```bash
[2019-02-02 22:42:52] [INFO] [MainThread] [model.py:29] loadding config from ./saves/config.pkl
[2019-02-02 22:42:54] [INFO] [MainThread] [model.py:102] loadding model from ./saves/model.pkl
[{'end': 12, 'entity': '框', 'start': 12, 'type': 'Thing'},
 {'end': 15, 'entity': '我们', 'start': 14, 'type': 'Person'}]
```

## 参考

- [sequence_tagging](https://github.com/AdolHong/sequence_tagging)
- [ChineseEmbedding](https://github.com/liuhuanyong/ChineseEmbedding)
- [Chinese-Literature-NER-RE-Dataset](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset)
- [a-PyTorch-Tutorial-to-Sequence-Labeling](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling)
- [chinese_text_cnn](https://github.com/bigboNed3/chinese_text_cnn)
- [torchtext](https://github.com/pytorch/text)
- [Torchtext 详细介绍](https://zhuanlan.zhihu.com/p/37223078)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
- [Pytorch中的RNN之pack_padded_sequence()和pad_packed_sequence()](https://www.cnblogs.com/sbj123456789/p/9834018.html)
- [python的Tqdm模块](https://blog.csdn.net/langb2014/article/details/54798823)
- [PyTorch 常用方法总结4：张量维度操作（拼接、维度扩展、压缩、转置、重复……）](https://zhuanlan.zhihu.com/p/31495102)