# lightNLP, lightsmile个人的自然语言处理框架

## 前言

依据自然语言处理四大任务等，框架主要设计为有以下五大功能：

- 序列标注， Sequence Labeling
- 文本分类， Text Classification
- 句子关系， Sentence Relation
- 文本生成， Text Generation
- 结构分析， Structure Parsing

因此将有五个主要的功能模块：sl（序列标注）、tc（文本分类）、sr（句子关系）、tg（文本生成）、sp（结构分析）和其他功能模块如we（字嵌入）。

## 当前已实现的功能：

### 序列标注，sl
- 中文分词，cws
- 命名实体识别，ner
- 词性标注，pos
- 语义角色标注， srl

### 结构分析，sp
- 基于图的依存句法分析，gdp
- 基于转移的依存句法分析， tdp

### 句子关系，sr
- 语句相似度，ss
- 文本蕴含，te

### 文本分类，tc
- 关系抽取，re
- 情感极性分析，sa

### 文本生成，tg
- 语言模型，lm


## 安装

本项目基于Pytorch1.0

```bash
pip install lightNLP
```

建议使用国内源来安装，如使用以下命令：
```bash
pip install -i https://pypi.douban.com/simple/ lightNLP
```

### 安装依赖

由于有些库如pytorch、torchtext并不在pypi源中或者里面只有比较老旧的版本，我们需要单独安装一些库。
#### 安装pytorch

具体安装参见[pytorch官网](https://pytorch.org/get-started/locally/)来根据平台、安装方式、Python版本、CUDA版本来选择适合自己的版本。

#### 安装torchtext

使用以下命令安装最新版本torchtext：
```bash
pip install https://github.com/pytorch/text/archive/master.zip
```



## 模型

- ner: BiLstm-Crf
- cws：BiLstm-Crf
- pos：BiLstm-Crf
- srl：BiLstm-Crf
- sa: TextCnn
- re：TextCnn,当前这里只是有监督关系抽取
- lm: Lstm,基础的LSTM，没有使用Seq2Seq模型
- ss: 共享LSTM + 曼哈顿距离
- te：共享LSTM + 全连接
- tdp：lstm + mlp + shift-reduce(移入规约)
- gdp：lstm + mlp + biaffine（双仿射）

## 训练数据标签

我这里仅是针对当前各任务从网上获取到的训练数据结构类型，有的形式可能并不规范或统一。

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

#### cws

BIS

训练数据示例如下：

```bash
4 S
日 S
清 B
晨 I
， S
同 B
样 I
在 S
安 B
新 I
县 I
人 B
民 I
政 I
府 I
门 B
前 I
， S
不 B
时 I
有 S
民 B
众 I
专 B
程 I
来 I
此 S
拍 B
照 I
留 B
念 I
， S
有 S
的 S
甚 B
至 I
穿 B
着 I
统 B
一 I
的 S
服 B
饰 I
拍 B
起 I
了 S
集 B
体 I
照 I
。 S
```

#### pos

BIS

训练数据示例如下：

```bash
只 B-c
要 I-c
我 B-r
们 I-r
进 B-d
一 I-d
步 I-d
解 B-i
放 I-i
思 I-i
想 I-i
， S-w
实 B-i
事 I-i
求 I-i
是 I-i
， S-w
抓 B-v
住 I-v
机 B-n
遇 I-n
， S-w
开 B-l
拓 I-l
进 I-l
取 I-l
， S-w
建 B-v
设 I-v
有 S-v
中 B-ns
国 I-ns
特 B-n
色 I-n
社 B-n
会 I-n
主 I-n
义 I-n
的 S-u
道 B-n
路 I-n
就 S-c
会 S-v
越 S-d
走 S-v
越 S-d
宽 B-a
广 I-a
。 S-w
```

#### srl

CONLL

训练数据示例如下，其中各列分别为`词`、`词性`、`是否语义谓词`、`角色`，每句仅有一个谓语动词为语义谓词，即每句中第三列仅有一行取值为1，其余都为0.

```bash
宋浩京  NR      0       O
转达    VV      0       O
了      AS      0       O
朝鲜    NR      0       O
领导人  NN      0       O
对      P       0       O
中国    NR      0       O
领导人  NN      0       O
的      DEG     0       O
亲切    JJ      0       O
问候    NN      0       O
，      PU      0       O
代表    VV      0       O
朝方    NN      0       O
对      P       0       O
中国    NR      0       B-ARG0
党政    NN      0       I-ARG0
领导人  NN      0       I-ARG0
和      CC      0       I-ARG0
人民    NN      0       E-ARG0
哀悼    VV      1       rel
金日成  NR      0       B-ARG1
主席    NN      0       I-ARG1
逝世    VV      0       E-ARG1
表示    VV      0       O
深切    JJ      0       O
谢意    NN      0       O
。      PU      0       O
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

#### re

训练数据示例如下，其中各列分别为`实体1`、`实体2`、`关系`、`句子`

```bash
钱钟书	辛笛	同门	与辛笛京沪唱和聽钱钟书与钱钟书是清华校友，钱钟书高辛笛两班。
元武	元华	unknown	于师傅在一次京剧表演中，选了元龙（洪金宝）、元楼（元奎）、元彪、成龙、元华、元武、元泰7人担任七小福的主角。
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

#### ss
tsv文件类型

训练数据示例如下，其中各列分别为`语句a`，`语句b`，`相似关系`，包括`0，不相似`，`1，相似`：
```bash
1       怎么更改花呗手机号码    我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号    1
2       也开不了花呗，就这样了？完事了  真的嘛？就是花呗付款    0
3       花呗冻结以后还能开通吗  我的条件可以开通花呗借款吗      0
4       如何得知关闭借呗        想永久关闭借呗  0
5       花呗扫码付钱    二维码扫描可以用花呗吗  0
6       花呗逾期后不能分期吗    我这个 逾期后还完了 最低还款 后 能分期吗        0
7       花呗分期清空    花呗分期查询    0
8       借呗逾期短信通知        如何购买花呗短信通知    0
9       借呗即将到期要还的账单还能分期吗        借呗要分期还，是吗      0
10      花呗为什么不能支付手机交易      花呗透支了为什么不可以继续用了  0
```


#### te
tsv文件类型

训练数据示例如下，其中各列分别为`前提`、`假设`、`关系`，其中关系包括`entailment，蕴含`、`neutral，中立`、`contradiction，矛盾`

```bash
是的，我想一个洞穴也会有这样的问题      我认为洞穴可能会有更严重的问题。        neutral
几周前我带他和一个朋友去看幼儿园警察    我还没看过幼儿园警察，但他看了。        contradiction
航空旅行的扩张开始了大众旅游的时代，希腊和爱琴海群岛成为北欧人逃离潮湿凉爽的夏天的令人兴奋的目的地。    航空旅行的扩大开始了许多旅游业的发展。  entailment
当两名工人待命时，一条大的白色管子正被放在拖车上。      这些人正在监督管道的装载。      neutral
男人俩互相交换一个很大的金属环，骑着火车向相反的方向行驶。      婚礼正在教堂举行。      contradiction
一个小男孩在秋千上玩。  小男孩玩秋千    entailment

```

#### tdp

格式大致如下, 其中每行代表一个`sentence`和对应的`Actions`，两者用` ||| `分隔，其中Actions包括三种：`Shift`、`REDUCE_R`和`REDUCE_L`，分别代表`移入`、`右规约`、`左规约`，其中sentence和Actions之间的序列长度对应关系为```len(Actions) = 2 * len(sentence) - 1``` ：

```bash
Bell , based in Los Angeles , makes and distributes electronic , computer and building products . ||| SHIFT SHIFT REDUCE_R SHIFT SHIFT SHIFT SHIFT REDUCE_L REDUCE_R REDUCE_R REDUCE_R SHIFT REDUCE_R SHIFT REDUCE_L SHIFT REDUCE_R SHIFT REDUCE_R SHIFT SHIFT REDUCE_R SHIFT REDUCE_R SHIFT REDUCE_R SHIFT REDUCE_R SHIFT REDUCE_L REDUCE_R SHIFT REDUCE_R
`` Apparently the commission did not really believe in this ideal . '' ||| SHIFT SHIFT SHIFT SHIFT REDUCE_L SHIFT SHIFT SHIFT SHIFT REDUCE_L REDUCE_L REDUCE_L REDUCE_L REDUCE_L REDUCE_L SHIFT SHIFT SHIFT REDUCE_L REDUCE_R REDUCE_R SHIFT REDUCE_R SHIFT REDUCE_R
```
#### gdp

CONLL格式，其中各列含义如下：

```bash
1	ID	当前词在句子中的序号，１开始.
2	FORM	当前词语或标点  
3	LEMMA	当前词语（或标点）的原型或词干，在中文中，此列与FORM相同
4	CPOSTAG	当前词语的词性（粗粒度）
5	POSTAG	当前词语的词性（细粒度）
6	FEATS	句法特征，在本次评测中，此列未被使用，全部以下划线代替。
7	HEAD	当前词语的中心词
8	DEPREL	当前词语与中心词的依存关系
```
 在CONLL格式中，每个词语占一行，无值列用下划线'_'代替，列的分隔符为制表符'\t'，行的分隔符为换行符'\n'；句子与句子之间用空行分隔。
 
 示例如：
 
 ```bash
1       坚决    坚决    a       ad      _       2       方式
2       惩治    惩治    v       v       _       0       核心成分
3       贪污    贪污    v       v       _       7       限定
4       贿赂    贿赂    n       n       _       3       连接依存
5       等      等      u       udeng   _       3       连接依存
6       经济    经济    n       n       _       7       限定
7       犯罪    犯罪    v       vn      _       2       受事

1       最高    最高    n       nt      _       3       限定
2       人民    人民    n       nt      _       3       限定
3       检察院  检察院  n       nt      _       4       限定
4       检察长  检察长  n       n       _       0       核心成分
5       张思卿  张思卿  n       nr      _       4       同位语
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

### cws

#### 训练

```python
from lightnlp.sl import CWS

cws_model = CWS()

train_path = '/home/lightsmile/NLP/corpus/cws/train.sample.txt'
dev_path = '/home/lightsmile/NLP/corpus/cws/test.sample.txt'
vec_path = '/home/lightsmile/NLP/embedding/char/token_vec_300.bin'

cws_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./cws_saves')
```

#### 测试

```python
cws_model.load('./cws_saves')

cws_model.test(dev_path)
```

#### 预测

```python
print(cws_model.predict('抗日战争时期，胡老在与侵华日军交战中四次负伤，是一位不折不扣的抗战老英雄'))
```

预测结果：

```bash
['抗日战争', '时期', '，', '胡老', '在', '与', '侵华日军', '交战', '中', '四次', '负伤', '，', '是', '一位', '不折不扣', '的', '抗战', '老', '英雄']
```

### pos

#### 训练

```python
from lightnlp.sl import POS

pos_model = POS()

train_path = '/home/lightsmile/NLP/corpus/pos/train.sample.txt'
dev_path = '/home/lightsmile/NLP/corpus/pos/test.sample.txt'
vec_path = '/home/lightsmile/NLP/embedding/char/token_vec_300.bin'

pos_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./pos_saves')
```

#### 测试

```python
pos_model.load('./pos_saves')

pos_model.test(dev_path)
```

#### 预测

```python
print(pos_model.predict('向全国各族人民致以诚挚的问候！'))
```

预测结果：

```bash
[('向', 'p'), ('全国', 'n'), ('各族', 'r'), ('人民', 'n'), ('致以', 'v'), ('诚挚', 'a'), ('的', 'u'), ('问候', 'vn'), ('！', 'w')]
```

### srl

#### 训练

```python
from lightnlp.sl import SRL

srl_model = SRL()

train_path = '/home/lightsmile/NLP/corpus/srl/train.sample.tsv'
dev_path = '/home/lightsmile/NLP/corpus/srl/test.sample.tsv'
vec_path = '/home/lightsmile/NLP/embedding/word/sgns.zhihu.bigram-char'


srl_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./srl_saves')
```

#### 测试

```python
srl_model.load('./srl_saves')

srl_model.test(dev_path)
```

#### 预测

```python
word_list = ['代表', '朝方', '对', '中国', '党政', '领导人', '和', '人民', '哀悼', '金日成', '主席', '逝世', '表示', '深切', '谢意', '。']
pos_list = ['VV', 'NN', 'P', 'NR', 'NN', 'NN', 'CC', 'NN', 'VV', 'NR', 'NN', 'VV', 'VV', 'JJ', 'NN', 'PU']
rel_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

print(srl_model.predict(word_list, pos_list, rel_list))
```

预测结果：

```bash
{'ARG0': '中国党政领导人和人民', 'rel': '哀悼', 'ARG1': '金日成主席逝世'}
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

### re

#### 训练

```python
from lightnlp.tc import RE

re = RE()

train_path = '/home/lightsmile/Projects/NLP/ChineseNRE/data/people-relation/train.sample.txt'
dev_path = '/home/lightsmile/Projects/NLP/ChineseNRE/data/people-relation/test.sample.txt'
vec_path = '/home/lightsmile/NLP/embedding/word/sgns.zhihu.bigram-char'

re.train(train_path, dev_path=dev_path, vectors_path=vec_path, save_path='./re_saves')

```

#### 测试

```python
re.load('./re_saves')
re.test(dev_path)
```

#### 预测

```python
print(re.predict('钱钟书', '辛笛', '与辛笛京沪唱和聽钱钟书与钱钟书是清华校友，钱钟书高辛笛两班。'))
```

预测结果：

```python
(0.7306928038597107, '同门') # return格式为（预测概率，预测标签）
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

#### 得到给定上文下，下一个字的topK候选集及其概率
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

#### 评估当前上文下，某一个字作为下一个字的可能性

```python
print(lm_model.next_word('要不是', '萧'))
```
结果：

```bash
0.006356663070619106
```

### ss

#### 训练

```python
from lightnlp.sr import SS

ss_model = SS()

train_path = '/home/lightsmile/Projects/NLP/sentence-similarity/input/atec/ss_train.tsv'
dev_path = '/home/lightsmile/Projects/NLP/sentence-similarity/input/atec/ss_dev.tsv'
vec_path = '/home/lightsmile/NLP/embedding/char/token_vec_300.bin'

ss_model.train(train_path, vectors_path=vec_path, dev_path=train_path, save_path='./ss_saves')
```

#### 测试

```python
ss_model.load('./ss_saves')
ss_model.test(dev_path)
```

#### 预测

```python
print(float(ss_model.predict('花呗更改绑定银行卡', '如何更换花呗绑定银行卡')))
```

预测结果：

```bash
0.9970847964286804
```

### te

#### 训练

```python
from lightnlp.sr import TE

te_model = TE()

train_path = '/home/lightsmile/Projects/liuhuaiyong/ChineseTextualInference/data/te_train.tsv'
dev_path = '/home/lightsmile/Projects/liuhuaiyong/ChineseTextualInference/data/te_dev.tsv'
vec_path = '/home/lightsmile/NLP/embedding/char/token_vec_300.bin'

te_model.train(train_path, vectors_path=vec_path, dev_path=train_path, save_path='./te_saves')
```

#### 测试

```python
te_model.load('./te_saves')
te_model.test(dev_path)
```

#### 预测

```python
print(te_model.predict('一个小男孩在秋千上玩。', '小男孩玩秋千'))
print(te_model.predict('两个年轻人用泡沫塑料杯子喝酒时做鬼脸。', '两个人在跳千斤顶。'))
```

预测结果为：

```bash
(0.4755808413028717, 'entailment')
(0.5721057653427124, 'contradiction')
```

### tdp

#### 训练

```python
from lightnlp.sp import TDP

tdp_model = TDP()

train_path = '/home/lightsmile/Projects/NLP/DeepDependencyParsingProblemSet/data/train.sample.txt'
dev_path = '/home/lightsmile/Projects/NLP/DeepDependencyParsingProblemSet/data/dev.txt'
vec_path = '/home/lightsmile/NLP/embedding/english/glove.6B.100d.txt'

tdp_model.train(train_path, dev_path=dev_path, vectors_path=vec_path,save_path='./tdp_saves')
```

#### 测试

```python
tdp_model.load('./tdp_saves')
tdp_model.test(dev_path)
```

#### 预测

```python
from pprint import pprint
pprint(tdp_model.predict('Investors who want to change the required timing should write their representatives '
                         'in Congress , he added . '))
```

预测结果如下：
```bash
{DepGraphEdge(head=(',', 14), modifier=('he', 15)),
 DepGraphEdge(head=('<ROOT>', -1), modifier=('Investors', 0)),
 DepGraphEdge(head=('Congress', 13), modifier=(',', 14)),
 DepGraphEdge(head=('Investors', 0), modifier=('who', 1)),
 DepGraphEdge(head=('he', 15), modifier=('added', 16)),
 DepGraphEdge(head=('in', 12), modifier=('Congress', 13)),
 DepGraphEdge(head=('representatives', 11), modifier=('in', 12)),
 DepGraphEdge(head=('required', 6), modifier=('timing', 7)),
 DepGraphEdge(head=('should', 8), modifier=('their', 10)),
 DepGraphEdge(head=('the', 5), modifier=('change', 4)),
 DepGraphEdge(head=('the', 5), modifier=('required', 6)),
 DepGraphEdge(head=('their', 10), modifier=('representatives', 11)),
 DepGraphEdge(head=('their', 10), modifier=('write', 9)),
 DepGraphEdge(head=('timing', 7), modifier=('should', 8)),
 DepGraphEdge(head=('to', 3), modifier=('the', 5)),
 DepGraphEdge(head=('want', 2), modifier=('to', 3)),
 DepGraphEdge(head=('who', 1), modifier=('want', 2))}
```

返回的格式类型为`set`，其中`DepGraphEdge`为命名元组，包含`head`和`modifier`两元素，这两元素都为`(word, position)`元组

### gdp

#### 训练

```python
from lightnlp.sp import GDP

gdp_model = GDP()

train_path = '/home/lightsmile/NLP/corpus/dependency_parse/THU/train.sample.conll'
vec_path = '/home/lightsmile/NLP/embedding/word/sgns.zhihu.bigram-char'


gdp_model.train(train_path, dev_path=train_path, vectors_path=vec_path, save_path='./gdp_saves')
```

#### 测试

```python
gdp_model.load('./gdp_saves')
gdp_model.test(train_path)
```

#### 预测

```python
word_list = ['最高', '人民', '检察院', '检察长', '张思卿']
pos_list = ['nt', 'nt', 'nt', 'n', 'nr']
heads, rels = gdp_model.predict(word_list, pos_list)
print(heads)
print(rels)
```

预测结果如下，其中程序会自动在语句和词性序列首部填充`<ROOT>`，因此返回的结果长度为`len(word_list) + 1`：
```bash
[0, 3, 3, 4, 0, 4]
['<ROOT>', '限定', '限定', '限定', '核心成分', '同位语']
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
    - cws，中文分词
    - pos，词性标注
    - srl，语义角色标注
- sp，结构分析
    - tdp，基于转移的依存句法分析
    - gdp，基于图的依存句法分析
- sr，句子关系
    - ss，句子相似度
    - te，文本蕴含
- tc，文本分类
    - re, 关系抽取
    - sa，情感分析
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

### 业务

### 工程

- [ ] 增加断点重训功能。
- [ ] 增加earlyStopping。
- [x] 重构项目结构，将相同冗余的地方合并起来，保持项目结构清晰
- [ ] 现在模型保存的路径和名字默认一致，会冲突，接下来每个模型都有自己的`name`。

### 功能

- [ ] 增加词向量相关模型以及训练预测代码
- [x] 增加情感分析相关模型以及训练预测代码
- [x] 增加文本蕴含相关模型以及训练预测代码
- [x] 增加文本生成相关模型以及训练预测代码
- [x] 增加语言模型相关模型以及训练预测代码
- [x] 增加依存分析相关模型以及训练预测代码
- [x] 增加关系抽取相关模型以及训练预测代码
- [x] 增加中文分词相关模型以及训练预测代码
- [x] 增加词性标注相关模型以及训练预测代码
- [x] 增加事件抽取相关模型以及训练预测代码
- [ ] 增加属性抽取相关模型以及训练预测代码
- [ ] 增加指代消解相关模型以及训练预测代码
- [ ] 增加自动摘要相关模型以及训练预测代码 
- [ ] 增加阅读理解相关模型以及训练预测代码
- [x] 增加句子相似度相关模型以及训练预测代码
- [ ] 增加序列到序列相关模型以及训练预测代码
- [ ] 增加关键词抽取相关模型以及训练预测代码
- [x] 增加命名实体识别相关模型以及预测训练代码

## 参考

### Deep Learning

- [What's the difference between “hidden” and “output” in PyTorch LSTM?](https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm)
- [What's the difference between LSTM() and LSTMCell()?](https://stackoverflow.com/questions/48187283/whats-the-difference-between-lstm-and-lstmcell)
- [深度学习框架技术剖析[转]](https://aiuai.cn/aifarm904.html)

### NLP

- [知识抽取-实体及关系抽取](http://www.shuang0420.com/2018/09/15/%E7%9F%A5%E8%AF%86%E6%8A%BD%E5%8F%96-%E5%AE%9E%E4%BD%93%E5%8F%8A%E5%85%B3%E7%B3%BB%E6%8A%BD%E5%8F%96/)
- [知识抽取-事件抽取](http://www.shuang0420.com/2018/10/15/%E7%9F%A5%E8%AF%86%E6%8A%BD%E5%8F%96-%E4%BA%8B%E4%BB%B6%E6%8A%BD%E5%8F%96/)

### Pytorch教程

- [PyTorch 常用方法总结4：张量维度操作（拼接、维度扩展、压缩、转置、重复……）](https://zhuanlan.zhihu.com/p/31495102)
- [Pytorch中的RNN之pack_padded_sequence()和pad_packed_sequence()](https://www.cnblogs.com/sbj123456789/p/9834018.html)
- [pytorch学习笔记（二）：gradient](https://blog.csdn.net/u012436149/article/details/54645162)
- [torch.multinomial()理解](https://blog.csdn.net/monchin/article/details/79787621)
- [Pytorch 细节记录](https://www.cnblogs.com/king-lps/p/8570021.html)
- [What does flatten_parameters() do?](https://stackoverflow.com/questions/53231571/what-does-flatten-parameters-do)
- [关于Pytorch的二维tensor的gather和scatter_操作用法分析](https://www.cnblogs.com/HongjianChen/p/9450987.html)
- [Pytorch scatter_ 理解轴的含义](https://blog.csdn.net/qq_16234613/article/details/79827006)
- [‘model.eval()’ vs ‘with torch.no_grad()’](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
- [到底什么是生成式对抗网络GAN？](https://www.msra.cn/zh-cn/news/features/gan-20170511)

### torchtext介绍

- [torchtext](https://github.com/pytorch/text)
- [A Tutorial on Torchtext](http://anie.me/On-Torchtext/)
- [Torchtext 详细介绍](https://zhuanlan.zhihu.com/p/37223078)
- [torchtext入门教程，轻松玩转文本数据处理](https://zhuanlan.zhihu.com/p/31139113)

### 其他工具模块

- [python的Tqdm模块](https://blog.csdn.net/langb2014/article/details/54798823)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)

### 词向量

- [ChineseEmbedding](https://github.com/liuhuanyong/ChineseEmbedding)

### 数据集

- [Chinese-Literature-NER-RE-Dataset](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset)
- [ChineseTextualInference](https://github.com/liuhuanyong/ChineseTextualInference)

### 序列标注

- [a-PyTorch-Tutorial-to-Sequence-Labeling](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling)
- [sequence_tagging](https://github.com/AdolHong/sequence_tagging)

### 文本分类

- [chinese_text_cnn](https://github.com/bigboNed3/chinese_text_cnn)

### 命名实体识别

- [sequence_tagging](https://github.com/AdolHong/sequence_tagging)

### 关系抽取

- [ChineseNRE](https://github.com/buppt/ChineseNRE)
- [pytorch-pcnn](https://github.com/ShomyLiu/pytorch-pcnn)
- [关系抽取(分类)总结](http://shomy.top/2018/02/28/relation-extraction/)

### 事件抽取

这里目前粗浅的将语义角色标注技术实现等同于事件抽取任务。

- [语义角色标注](http://wiki.jikexueyuan.com/project/deep-learning/wordSence-identify.html)
- [iobes_iob 与 iob_ranges 函数借鉴](https://github.com/glample/tagger/blob/master/utils.py)
- [BiRNN-SRL](https://github.com/zxplkyy/BiRNN-SRL)
- [chinese_semantic_role_labeling](https://github.com/Nrgeup/chinese_semantic_role_labeling)

### 语言模型

- [char-rnn.pytorch](https://github.com/spro/char-rnn.pytorch)
- [Simple Word-based Language Model in PyTorch](https://github.com/deeplearningathome/pytorch-language-model)
- [PyTorch 中级篇（5）：语言模型（Language Model (RNN-LM)）](https://shenxiaohai.me/2018/10/20/pytorch-tutorial-intermediate-05/)

### 文本生成

- [好玩的文本生成](https://www.msra.cn/zh-cn/news/features/ruihua-song-20161226)
- [基于深度学习的文本生成过程](https://puke3615.github.io/2018/08/10/ML-Text-Generator/)

### 语句相似度

- [siamese_lstm](https://github.com/WEAINE/siamese_lstm)
- [sentence-similarity](https://github.com/yanqiangmiffy/sentence-similarity)


### 文本蕴含

- [ChineseTextualInference](https://github.com/liuhuanyong/ChineseTextualInference)

### 中文分词
- [中文自然语言处理中文分词训练语料](https://download.csdn.net/download/qq_36330643/10514771)
- [中文分词、词性标注联合模型](https://zhuanlan.zhihu.com/p/56988686)
- [pytorch_Joint-Word-Segmentation-and-POS-Tagging](https://github.com/bamtercelboo/pytorch_Joint-Word-Segmentation-and-POS-Tagging)

### 词性标注

- [常见中文词性标注集整理](https://blog.csdn.net/qq_41853758/article/details/82924325)
- [分词：词性标注北大标准](https://blog.csdn.net/zhoubl668/article/details/6942251)
- [ICTCLAS 汉语词性标注集 中科院](https://blog.csdn.net/memray/article/details/14105643)
- [中文文本语料库整理](https://www.jianshu.com/p/206caa232ded)
- [中文分词、词性标注联合模型](https://zhuanlan.zhihu.com/p/56988686)
- [pytorch_Joint-Word-Segmentation-and-POS-Tagging](https://github.com/bamtercelboo/pytorch_Joint-Word-Segmentation-and-POS-Tagging)

### 指代消解

- [AllenNLP系列文章之四：指代消解](https://blog.csdn.net/sparkexpert/article/details/79868335)

### 依存句法分析

- [汉语树库](http://www.hankcs.com/nlp/corpus/chinese-treebank.html#h3-6)
- [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
- [中文句法结构](https://xiaoxiaoaurora.github.io/2018/07/03/%E4%B8%AD%E6%96%87%E5%8F%A5%E6%B3%95%E7%BB%93%E6%9E%84/)
- [句法分析之依存句法](https://nlpcs.com/article/syntactic-parsing-by-dependency)
- [Deep Biaffine Attention for Neural Dependency Parsing, hankcs简要解读](http://www.hankcs.com/nlp/parsing/deep-biaffine-attention-for-neural-dependency-parsing.html)
- [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198)
- [biaffine-parser](https://github.com/zysite/biaffine-parser)
- [DeepDependencyParsingProblemSet](https://github.com/rguthrie3/DeepDependencyParsingProblemSet)

### 自动摘要

- [干货｜当深度学习遇见自动文本摘要，seq2seq+attention](https://blog.csdn.net/Mbx8X9u/article/details/80491214)

### 阅读理解

- [ASReader：一个经典的机器阅读理解深度学习模型](https://www.imooc.com/article/28709)

### 其他

- [基于距离的算法 曼哈顿，欧氏等](https://www.jianshu.com/p/bbe6dfac9bc7)
- [在分类中如何处理训练集中不平衡问题](https://blog.csdn.net/heyongluoyao8/article/details/49408131)
- [Python-Pandas 如何shuffle（打乱）数据？](https://blog.csdn.net/qq_22238533/article/details/70917102)
- [Python DataFrame 如何删除原来的索引，重新建立索引](https://www.cnblogs.com/xubing-613/p/6119162.html)
- [Pandas在读取csv时如何设置列名--常用方法集锦](https://zhuanlan.zhihu.com/p/44503744)
- [Python中__repr__和__str__区别](https://blog.csdn.net/luckytanggu/article/details/53649156)

