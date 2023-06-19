# skill_NER_Chinese
基于中文语料库训练的技能实体识别模型。

标注过程中，倾向于硬技能（如C++、深度学习），而非软技能（团结合作、责任心）

模型：roberta_bilstm_crf

参考实现：https://github.com/taishan1994/BERT-BILSTM-CRF

# 测试

已训练的pytorch_model_ner.bin下载：链接：https://share.weiyun.com/v6o317Bj 密码：y4y2dr

放到./checkpoint/skill下。

# 训练

前往https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main下载config.json, pytorch_model.bin, vocab.txt

放到./model_hub/chinese-roberta-wwm-ext下。

# 依赖

```python
python==3.9.16
scikit-learn==1.1.3 
scipy==1.10.1 
seqeval==1.2.2
transformers==4.27.4
pytorch-crf==0.7.2
```

# 目录结构

```python
--checkpoint：模型和配置保存位置
----skill:
--------vocab.txt
--------pytorch_model_ner.bin
--------ner_args.json
--------config.json
--model_hub：预训练模型
----chinese-roberta-wwm-ext:
--------vocab.txt
--------pytorch_model.bin
--------config.json
--data：存放数据
----skill
--------ner_data：处理之后的数据
------------labels.txt：标签
------------train.txt：训练数据
------------dev.txt：测试数据
--config.py：配置
--model.py：模型
--predict.py：加载训练好的模型进行预测
--main.py：训练和测试
```

# 说明

```python

在config.py里面定义一些参数，比如：
--max_seq_len：句子最大长度，GPU显存不够则调小。
--epochs：训练的epoch数
--train_batch_size：训练的batchsize大小，GPU显存不够则调小。
--dev_batch_size：验证的batchsize大小，GPU显存不够则调小。
--save_step：多少step保存模型
其余的可保持不变。

1. main.py里定义了data_name为skill，可以修改为其他数据集。运行：python main.py

2. 命令行直接输入待抽取语句作为参数并运行，如：python predict.py 熟悉常用中间件，包括但不限于Redis、Kafka、Zookeeper等，了解其性能调优;
```

## skill数据集
```python
              precision    recall  f1-score   support

        技能       0.82667      0.86111      0.84354       648

文本>>>>>： 5、熟悉常用中间件，包括但不限于Redis、Kafka、Zookeeper等，了解其性能调优;
实体>>>>>： [{"span": "中间件", "offset": [6, 8], "tag": "技能"}, {"span": "Redis", "offset": [16, 20], "tag": "技能"}, {"span": "Kafka", "offset": [22, 26], "tag": "技能"}, {"span": "Zookeeper", "offset": [28, 36], "tag": "技能"}, {"span": "性能调优", "offset": [42, 45], "tag": "技能"}]
IOB>>>>>:   {"text": ["5", "、", "熟", "悉", "常", "用", "中", "间", "件", "，", "包", "括", "但", "不", "限", "于", "R", "e", "d", "i", "s", "、", "K", "a", "f", "k", "a", "、", "Z", "o", "o", "k", "e", "e", "p", "e", "r", "等", "，", "了", "解", "其", "性", "能", "调", "优", "；"], "labels": ["O", "O", "O", "O", "O", "O", "B-技能", "I-技能", "I-技能", "O", "O", "O", "O", "O", "O", "O", "B-技能", "I-技能", "I-技能", "I-技能", "I-技能", "O", "B-技能", "I-技能", "I-技能", "I-技能", "I-技能", "O", "B-技能", "I-技能", "I-技能", "I-技能", "I-技能", "I-技能", "I-技能", "I-技能", "I-技能", "O", "O", "O", "O", "O", "B-技能", "I-技能", "I-技能", "I-技能", "O"], "id": 521}
====================================================================================================
