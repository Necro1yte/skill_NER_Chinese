import os
import json
import torch
import numpy as np
import argparse

from collections import namedtuple
from model import BertNer
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer


def get_args(args_path, args_name=None):
    with open(args_path, "r") as fp:
        args_dict = json.load(fp)
    # 注意args不可被修改了
    args = namedtuple(args_name, args_dict.keys())(*args_dict.values())
    return args


class Predictor:
    def __init__(self, data_name):
        self.data_name = data_name
        self.ner_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "ner_args.json"), "ner_args")
        self.ner_id2label = {int(k): v for k, v in self.ner_args.id2label.items()}
#        training stage
#        self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_pretrained_dir)
#        testing stage
        self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_ner_dir)
        self.max_seq_len = self.ner_args.max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_model = BertNer(self.ner_args)
        self.ner_model.load_state_dict(torch.load(os.path.join(self.ner_args.output_dir, "pytorch_model_ner.bin")))
        self.ner_model.to(self.device)
        self.data_name = data_name

    def ner_tokenizer(self, text):
        # print("文本长度需要小于：{}".format(self.max_seq_len))
        text = text[:self.max_seq_len - 2]
        text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = torch.tensor(np.array([input_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask

    def ner_predict(self, text):
        input_ids, attention_mask = self.ner_tokenizer(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.ner_model(input_ids, attention_mask)
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])
        logits = output.logits
        logits = logits[0][1:length - 1]
        logits = [self.ner_id2label[i] for i in logits]
        entities = get_entities(logits)
        result = {}
        for ent in entities:
            ent_name = ent[0]
            ent_start = ent[1]
            ent_end = ent[2]
            if ent_name not in result:
                result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
            else:
                result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
        return result


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='命令行中输入的文本将被用于实体抽取。')
        parser.add_argument('input_text', type=str, help='输入的文本')
        args = parser.parse_args()

        data_name = "skill"
        predictor = Predictor(data_name)
        ner_result = predictor.ner_predict(args.input_text)
    
        print("文本>>>>>：", args.input_text)
        print("实体>>>>>：", ner_result)
        print("="*100)


