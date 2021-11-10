# e2e-joint-coref

Tackling Zero Pronoun Resolution and Non-Zero Coreference Resolution Jointly

## 目录结构

+ data/: 数据目录，用于存放 ontonotes、train/test/val、checkpoint
+ config.py: 配置文件
+ train.py: 训练文件
+ tools.py: 工具文件
+ model.py: 模型文件
+ onf_to_data.py: ontonotes数据集处理文件

## 要求配置

### 硬件需求

程序需要 32G 显存，训练时可通过修改 config.py 中的 max_training_sentences 等配置降低显存使用

### python依赖包

+ python
+ torch
+ transformers
+ numpy


## 数据集收集和处理

### 使用 ontonotes 数据集

1. 从 LDC 网站下载 ontonotes 数据集，若做中文零指代，将 config.py 中的 "ontonotes_root_dir" 修改为 "/path/to/ontonotes/data/files/data/chinese/annotations"
2. 运行命令 ```python onf_to_data.py```
3. 若没出现问题，则 data/ 下会生成 train.json/test.json/val.json 三个文件

### 使用自己的数据

将数据根据需要分割成 train.json/test.json/val.json

三个json文件的格式为：每行代表一个文档，每行都是一个json对象，json对象具体内容为：（注：此处为了美观将json对象格式化为多行，在真实文件中需为1行）

```
{
    "doc_key": "文档的地址", 
    "sentences": [["token1", "token2", ...], ...],
    "clusters": [[[sloc1, eloc1], [sloc2, eloc2], ...], ...],
    "speaker_ids" [["speaker#1", ...], ...]
    "sentence_map": [[0, 0, 0, ..., 3, 3, 3], ...],
    "subtoken_map": [[0, 0, 1, 2, 3, ...], ...]
}
```

说明：sentences中一个元素代表一个长句子，可由若干个短句子组成，使用onf_to_data.py生成的长句子长度在 max_seq_length 附近，max_seq_length 可在 config.py 文件中设置；sentences中组成长句子的若干个短句子由 **tokenize 后的 token** 组成；clusters 是文档中所有指代链的集合，由若干指代链组成，一个指代链由若干个mention/零指代组成，mention使用位置表示：`[在文档中的token起始位置, 在文档中的token结束位置 + 1]`，零指代使用位置表示：`[零指代后一个token位置, 零指代后一个token位置]`。

例子（注：此处为了美观将json对象格式化为多行，在真实文件中每个文档需为1行）：


```
{
    "sentences": [["打", "雷", "了", "怎", "么", "发", "短", "信", "安", "慰", "女", "朋", "友", "？", "打", "雷", "时", "还", "给", "她", "发", "？"]],
    "clusters": [[[10, 13], [19, 20]], [[6, 8], [21, 21]]],         # （女朋友， 她）（短信，发与？间的零指代）
    "speaker_ids": [["a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b", "b", "b"]],
    "sentence_map": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]],
    "subtoken_map": [[1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16]],
    "genre": "dummy_genre",
    "doc_key": "dummy_data"
}
```

## 模型训练

1. 配置 config.py 中相关参数
2. 运行 ```python train.py```


## 模型测试

1. 配置 config.py 相关参数，将 evaluate.py 中的数据部分修改为希望测试的数据
2. 运行 ```pyhton evaluate.py```
