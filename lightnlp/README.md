# lightNLP, lightsmile个人的自然语言处理框架

## 前言

依据自然语言处理四大任务，框架主要设计为有以下功能：

- 序列标注， Sequence Labeling
- 文本分类， Text Classification
- 句子关系， Sentence Relation
- 文本生成， Text Generation

因此将有四个主要的功能模块：sl（序列标注）、tc（文本分类）、sr（句子关系）、tg（文本生成）和其他功能模块如we（字嵌入）。

当前只实现了sl下的命名实体识别（ner）功能和tc下的情感极性分析（sa）功能。

## 安装

本项目基于Pytorch1.0

```bash
pip install lightNLP
```

## 模型

- ner: BiLstm-Crf
- sa: TextCnn

## 训练数据标签

#### ner

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


#### sa

tsv文件格式

训练数据示例如下：

```bash
        label   text
0       0       备胎是硬伤！
1       0       要说不满意的话，那就是动力了，1.5自然吸气发动机对这款车有种小马拉大车的感觉。如今天气这么热，上路肯定得开空调，开了后动力明显感觉有些不给力不过空调制冷效果还是不错的。
2       0       油耗显示13升还多一点，希望慢慢下降。没有倒车雷达真可恨
3       0       空调不太凉，应该是小问题。
4       0       1、后排座椅不能平放；2、科技感不强，还不如百万帝豪，最希望增加车联网的车机。像你好博越一样。3、全景摄像头不清楚，晚上基本上用处不大
5       1       车子外观好看，车内空间大。
6       1       最满意的真的不只一点，概括一下最满意的就是性价比了。ps:虽然没有s7性价比高(原厂记录仪,绿净)
7       0       底盘调教的很低，坐的感觉有些别扭，视角不是很好。
8       0       开空调时，一档起步动力不足。车子做工有点马虎。
```
## 使用

### ner

#### 训练

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

#### 测试

```python
# 加载模型，默认当前目录下的`saves`目录
ner_model.load()
# 对train_path下的测试集进行读取测试
ner_model.test(train_path)
```

#### 预测

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

### sa

#### 训练

```python

from lightnlp.tc import SA

# 创建SA对象
sa_model = SA()

train_path = '/home/lightsmile/Projects/NLP/chinese_text_cnn/data/train.sample.tsv'
dev_path = '/home/lightsmile/Projects/NLP/chinese_text_cnn/data/dev.sample.tsv'
vec_path = '/home/lightsmile/Downloads/1410356697_浅笑哥fight/自然语言处理/词向量/sgns.zhihu.bigram-char'

# 只需指定训练数据路径，预训练字向量可选，开发集路径可选，模型保存路径可选。
sa_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./sa_saves')
```

#### 测试

```python

# 加载模型，默认当前目录下的`saves`目录
sa_model.load()

# 对train_path下的测试集进行读取测试
sa_model.test(train_path)
```

#### 预测

```python

sa_model.load()

from pprint import pprint

pprint(sa_model.predict('外观漂亮，安全性佳，动力够强，油耗够低'))
```

预测结果：

```python
(1.0, '1') # return格式为（预测概率，预测标签）
```

## todo

- 现在模型保存的路径和名字默认一致，会冲突，接下来每个模型都有自己的`name`。
- 增加earlyStopping。
- 增加断点重训功能。
- 重构项目结构，将相同冗余的地方合并起来，保持项目结构清晰
- 增加句子关系模型以及训练预测代码
- 增加文本生成模型以及训练预测代码
- 增加词向量相关模型以及训练预测代码
- 增加语言模型相关模型以及训练预测代码
- 增加关系抽取相关模型以及训练预测代码
- 增加事件抽取相关模型以及训练预测代码
- 增加属性抽取相关模型以及训练预测代码
- 增加依存分析相关模型以及训练预测代码
- 增加关键词抽取相关模型以及训练预测代码
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