# lightNLP, lightsmile个人的自然语言处理框架

## 前言

依据自然语言处理四大任务，框架主要设计为有以下功能：

- 序列标注， Sequence Labeling
- 文本分类， Text Classification
- 句子关系， Sentence Relation
- 文本生成， Text Generation

因此将有四个主要的功能模块：sl（序列标注）、tc（文本分类）、sr（句子关系）、tg（文本生成）和其他功能模块如we（字嵌入）。

当前已实现了sl下的命名实体识别（ner）功能、tc下的情感极性分析（sa）功能和tg下的语言模型（lm）功能。

## 安装

本项目基于Pytorch1.0

```bash
pip install lightNLP
```

## 模型

- ner: BiLstm-Crf
- sa: TextCnn
- lm: Lstm,基础的LSTM，没有使用Seq2Seq模型

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

#### lm
就普通的文本格式

训练数据示例如下：
```bash
第一章 陨落的天才
    
    “斗之力，三段！”
    望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛……
    “萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来……
    中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。
    “三段？嘿嘿，果然不出我所料，这个“天才”这一年又是在原地踏步！”
    “哎，这废物真是把家族的脸都给丢光了。”
    “要不是族长是他的父亲，这种废物，早就被驱赶出家族，任其自生自灭了，哪还有机会待在家族中白吃白喝。”
    “唉，昔年那名闻乌坦城的天才少年，如今怎么落魄成这般模样了啊？”

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
ner_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./ner_saves')
```

#### 测试

```python
# 加载模型，默认当前目录下的`saves`目录
ner_model.load('./ner_saves')
# 对train_path下的测试集进行读取测试
ner_model.test(train_path)
```

#### 预测

```python
# 加载模型，默认当前目录下的`saves`目录
ner_model.load('./ner_saves')

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
sa_model.load('./sa_saves')

# 对train_path下的测试集进行读取测试
sa_model.test(train_path)
```

#### 预测

```python

sa_model.load('./sa_saves')

from pprint import pprint

pprint(sa_model.predict('外观漂亮，安全性佳，动力够强，油耗够低'))
```

预测结果：

```python
(1.0, '1') # return格式为（预测概率，预测标签）
```

### lm
#### 训练

```python
from lightnlp.tg import LM

lm_model = LM()

train_path = '/home/lightsmile/NLP/corpus/lm_test.txt'
dev_path = '/home/lightsmile/NLP/corpus/lm_test.txt'
vec_path = '/home/lightsmile/NLP/embedding/char/token_vec_300.bin'

lm_model.train(train_path, vectors_path=vec_path, dev_path=train_path, save_path='./lm_saves')
```
#### 测试
```python
lm_model.load('./lm_saves')

lm_model.test(dev_path)
```
#### 预测

##### 文本生成
默认生成30个

```python
print(lm_model.generate_sentence('少年面无表情，唇角有着一抹自嘲'))
```
结果：

```bash
少年面无表情，唇角有着一抹自嘲，紧握的手掌，因，无所谓的面上，那抹讥讽所莫下了脚步，当时的
```

#### 得到个下一个词topK候选集及其概率
默认输出top5个

```python
print(lm_model.next_word_topk('少年面无表情，唇角'))
```

结果：

```bash
[('有', 0.9791949987411499), ('一', 0.006628755945712328), ('不', 0.004853296559303999), ('出', 0.0026260288432240486), ('狠', 0.0017451468156650662)]
```

#### 评估语句分数
结果为以10为底的对数，即`log10(x)`

```python
print(lm_model.sentence_score('少年面无表情，唇角有着一抹自嘲'))
```
结果：

```bash
-11.04862759023672
```

#### 评估当前上文下，某一个词作为下一个词的可能性

```python
print(lm_model.next_word('要不是', '萧'))
```
结果：

```bash
0.006356663070619106
```

## 项目组织结构
### 项目架构
- base
    - config.py
    - model.py
    - module.py
    - tool.py
- sl，序列标注
    - ner，命名实体识别
- sr，句子关系
    - ss，句子相似度
    - te，文本蕴含
- tc，文本分类
    -sa，情感分析
- tg，文本生成
    - lm，语言模型
    - mt，机器翻译
- utils
### 架构说明
#### base目录
放一些基础的模块实现，其他的高层业务模型以及相关训练代码都从此module继承相应父类。
##### config
存放模型训练相关的超参数等配置信息
##### model
模型的实现抽象基类，包含`base.model.BaseConfig`和`base.model.BaseModel`，包含`load`、`save`等方法
##### module
业务模块的训练验证测试等实现抽象基类，包含`base.module.Module`，包含`train`、`load`、`_validate`、`test`等方法
##### tool
业务模块的数据处理抽象基类，包含`base.tool.Tool`，包含`get_dataset`、`get_vectors`、`get_vocab`、`get_iterator`、`get_score`等方法
#### util目录
放一些通用的方法

## todo

- [ ] 现在模型保存的路径和名字默认一致，会冲突，接下来每个模型都有自己的`name`。
- [ ] 增加earlyStopping。
- [ ] 增加断点重训功能。
- [x] 重构项目结构，将相同冗余的地方合并起来，保持项目结构清晰
- [ ] 增加句子关系模型以及训练预测代码
- [ ] 增加文本生成模型以及训练预测代码
- [ ] 增加词向量相关模型以及训练预测代码
- [x] 增加语言模型相关模型以及训练预测代码
- [ ] 增加关系抽取相关模型以及训练预测代码
- [ ] 增加事件抽取相关模型以及训练预测代码
- [ ] 增加属性抽取相关模型以及训练预测代码
- [ ] 增加依存分析相关模型以及训练预测代码
- [ ] 增加关键词抽取相关模型以及训练预测代码

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