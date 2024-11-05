import argparse
import json
import os
import pickle

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


class REDataset(Dataset):
    def __init__(self, args, processer, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.data_items = processer.convert_to_bert_features(
            tokenizer) if args.use_bert else processer.convert_to_glove_features()

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data_items[idx]['input_ids']) if not isinstance(self.data_items[idx]['input_ids'],
                                                                                      torch.Tensor) else \
        self.data_items[idx]['input_ids']
        relation_id = torch.tensor(self.data_items[idx]['relation'])
        dep_adj = self.data_items[idx]['padding_dep_matrix']
        ctx_adj = self.data_items[idx]['padding_context_matrix']
        relative_subj = torch.tensor(self.data_items[idx]['relative_subj'])
        relative_obj = torch.tensor(self.data_items[idx]['relative_obj'])
        pos_ids = torch.tensor(self.data_items[idx]['pos_ids'])
        dep_ids = torch.tensor(self.data_items[idx]['dep_ids'])
        if self.args.use_bert:
            attention_mask = torch.tensor(self.data_items[idx]['attention_mask'])
            token_type_ids = torch.tensor(self.data_items[idx]['token_type_ids'])
            valid_ids = torch.tensor(self.data_items[idx]['valid_ids'])
            return input_ids, attention_mask, token_type_ids, valid_ids, relation_id, dep_adj, relative_subj, relative_obj, pos_ids, dep_ids, ctx_adj
        return input_ids, relation_id, dep_adj, relative_subj, relative_obj, pos_ids, dep_ids, ctx_adj

    def get_num_relations(self):
        data_dir = os.path.join(self.args.data_dir, self.args.task_name, 'relations.json')
        if not os.path.exists(data_dir):
            raise FileNotFoundError("Please generate relations.json first")
        with open(data_dir, 'r') as f:
            relations = json.load(f)
            return len(relations)


class REProcessor:
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

    def read_json_file(self, data_dir):
        with open(data_dir, 'r') as f:
            items = json.load(f)
            return items

    def get_deps_and_relations(self):

        dep_file_dir = os.path.join(self.args.data_dir, self.args.task_name, "dep_type.json")
        rel_file_dir = os.path.join(self.args.data_dir, self.args.task_name, "relations.json")
        if not os.path.exists(dep_file_dir) or not os.path.exists(rel_file_dir):
            mode = ['train', 'dev', 'test']
            dep_set = set()
            relation_set = set()
            for mode in mode:
                data_dir = os.path.join(self.args.data_dir, self.args.task_name, mode + ".json")
                if not os.path.exists(data_dir):
                    continue
                items = self.read_json_file(data_dir)
                for item in items:
                    dep_set.update(item['stanford_deprel'])
                    relation_set.add(item['relation'])
            dep_set.add("none")
            dep_dict = {dep: index for index, dep in enumerate(dep_set)}
            rel_list = {rel: index for index, rel in enumerate(relation_set)}
            with open(os.path.join(self.args.data_dir, self.args.task_name, "dep_type.json"), 'w') as dep, \
                    open(os.path.join(self.args.data_dir, self.args.task_name, "relations.json"), 'w') as rel:
                json.dump(dep_dict, dep)
                json.dump(rel_list, rel)
            return dep_dict, rel_list

        with open(dep_file_dir, 'r') as dep_dict, open(rel_file_dir, 'r') as rel_dict:
            dep_dict = json.load(dep_dict)
            rel_dict = json.load(rel_dict)
            return dep_dict, rel_dict

    def generate_dep_matrix(self, stanford_head):
        init_matrix = torch.zeros(len(stanford_head), len(stanford_head))
        for index in range(len(stanford_head)):
            init_matrix[index][index] = 1
            temp_num = int(stanford_head[index]) - 1
            if temp_num != -1:
                init_matrix[index][temp_num] = 1
                init_matrix[temp_num][index] = 1
        return init_matrix

    def process_data(self, data_items):
        processed_data_items = []
        for item in data_items:
            if "subj_type" in item:
                temp_subj = "SUBJ-" + item["subj_type"]
                temp_obj = "OBJ-" + item["obj_type"]
            else:
                temp_subj = "SUBJ"
                temp_obj = "OBJ"
            item["token"][item["subj_start"]: item["subj_end"] + 1] = [temp_subj]
            item["token"][item["obj_start"]: item["obj_end"] + 1] = [temp_obj]
            processed_data_items.append(item)
        return processed_data_items

    def build_glove_embedding(self):
        train_file = self.args.data_dir + self.args.task_name + "/train.json"
        dev_file = self.args.data_dir + self.args.task_name + "/dev.json"
        test_file = self.args.data_dir + self.args.task_name + "/test.json"
        glove_file = self.args.glove_dir
        glove_embedding_dim = self.args.glove_dim

        if not os.path.exists(self.args.vocab_dir + self.args.task_name):
            os.makedirs(self.args.vocab_dir + self.args.task_name)

        # loading glove
        print("loading glove")
        glove_vocab = []
        with open(glove_file, 'r') as gf:
            for line in gf:
                elems = line.split()
                glove_token = ''.join(elems[0:-glove_embedding_dim])
                glove_vocab.append(glove_token)

        print("loading data files")
        train_tokens = load_tokens(train_file)
        test_tokens = load_tokens(test_file)
        if os.path.exists(dev_file):
            dev_tokens = load_tokens(dev_file)
            global_vocabs = set(train_tokens + dev_tokens + test_tokens)
        else:
            global_vocabs = set(train_tokens + test_tokens)
        global_vocabs = list(global_vocabs)
        global_vocabs.extend(["PAD_TOKEN", "UNK_TOKEN"])
        print("writing glove vocab&embedding")
        emb = np.random.uniform(-1, 1, (len(global_vocabs), glove_embedding_dim))
        w2id = {t: i for i, t in enumerate(global_vocabs)}
        emb[w2id["PAD_TOKEN"]] = np.zeros(glove_embedding_dim)
        with open(glove_file, 'r') as gf:
            for line in gf:
                elems = line.split()
                token = ''.join(elems[0:-glove_embedding_dim])
                if token in w2id:
                    emb[w2id[token]] = [float(v) for v in elems[-glove_embedding_dim:]]

        print("embedding size:{} x {}".format(*emb.shape))
        vocab_save_path = os.path.join(self.args.vocab_dir, self.args.task_name, "vocab.pkl")
        emb_save_path = os.path.join(self.args.vocab_dir, self.args.task_name, "embedding.npy")
        with open(vocab_save_path, 'wb') as v:
            pickle.dump(global_vocabs, v)
        np.save(emb_save_path, emb)
        print("ALL DONE")

        print("saving glove vocab&embedding")

    def get_POS(self):
        mode = ['train', 'dev', 'test']
        data_dir = self.args.data_dir + self.args.task_name
        pos_file_dir = os.path.join(data_dir, "pos.json")
        if not os.path.exists(pos_file_dir):
            POS_set = set()
            for mode in mode:
                with open(os.path.join(data_dir, mode + ".json"), "r") as f:
                    data_items = json.load(f)
                    for item in data_items:
                        POS_set.update(item['stanford_pos'])
            POS_set.add("none")
            POS_list = list(POS_set)
            POS_dict = {k: i for i, k in enumerate(POS_list)}
            with open(pos_file_dir, 'w') as pos:
                json.dump(POS_dict, pos)
        else:
            with open(pos_file_dir, 'r') as pos:
                pos_dict = json.load(pos)
                return pos_dict

    def get_pos_dep_rel_len(self):
        pos_len = len(self.get_POS())
        dep_dict, rel_dict = self.get_deps_and_relations()
        return pos_len, len(dep_dict), len(rel_dict)

    def compute_relative_distance(self, tokens_len, e1_pos, e2_pos):
        relative_e1_pos = []
        relative_e2_pos = []
        for index in range(tokens_len):
            if index < e1_pos[0]:
                relative_e1_pos.append(e1_pos[0] - index)
            elif index > e1_pos[-1]:
                relative_e1_pos.append(index - e1_pos[-1])
            else:
                relative_e1_pos.append(0)
            if index < e2_pos[0]:
                relative_e2_pos.append(e2_pos[0] - index)
            elif index > e2_pos[-1]:
                relative_e2_pos.append(index - e2_pos[-1])
            else:
                relative_e2_pos.append(0)
        return relative_e1_pos, relative_e2_pos

    def compute_relative_distance_2(self, tokens_len, subj_pos, obj_pos):
        relative_subj = []
        relative_obj = []
        for i in range(tokens_len):
            if subj_pos[0] <= i <= subj_pos[1]:
                relative_subj.append(0)
            else:
                relative_subj.append((i-subj_pos[0]) if i < subj_pos[0] else (i - subj_pos[1]))
            if obj_pos[0] <= i <= obj_pos[1]:
                relative_obj.append(0)
            else:
                relative_obj.append((i-obj_pos[0]) if i < obj_pos[0] else (i - obj_pos[1]))
        max_subj = max(np.abs(relative_subj))
        max_obj = max(np.abs(relative_obj))
        relative_subj += max_subj
        relative_obj += max_obj
        return relative_subj.tolist(), relative_obj.tolist()

    def convert_to_bert_features(self, tokenizer):
        features = []
        dep_dict, rel_dict = self.get_deps_and_relations()
        pos_dict = self.get_POS()
        data_dir = os.path.join(self.args.data_dir, self.args.task_name, self.mode + ".json")
        items = self.process_data(self.read_json_file(data_dir))
        for index, item in enumerate(items):
            tokens = ["[CLS]"]
            valid = [0]
            subj_pos = [item["subj_start"], item["subj_end"]]
            obj_pos = [item["obj_start"], item["obj_end"]]
            relative_subj, relative_subj = self.compute_relative_distance(len(item["token"]), subj_pos, obj_pos)
            for i, token in enumerate(item["token"]):
                token = tokenizer.tokenize(token)
                tokens.extend(token)
                for j in range(len(token)):
                    if j == 0:
                        valid.append(1)
                    else:
                        valid.append(0)
                if len(tokens) >= self.args.max_seq_len - 1:
                    break
            tokens.append("[SEP]")
            valid.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            dep_matrix = self.generate_dep_matrix(item["stanford_head"])
            pos_ids = [pos_dict[p] for p in item["stanford_pos"]]
            dep_ids = [dep_dict[d] for d in item["stanford_deprel"]]

            input_ids += [0] * (self.args.max_seq_len - len(input_ids))
            attention_mask += [0] * (self.args.max_seq_len - len(attention_mask))
            token_type_ids += [0] * (self.args.max_seq_len - len(token_type_ids))
            valid += [0] * (self.args.max_seq_len - len(valid))
            relation_id = rel_dict[item["relation"]]
            pos_ids += [pos_dict["none"]] * (self.args.max_seq_len - len(pos_ids))
            dep_ids += [dep_dict["none"]] * (self.args.max_seq_len - len(dep_ids))
            padding_dep_matrix = torch.zeros(self.args.max_seq_len, self.args.max_seq_len)
            for i in range(len(dep_matrix)):
                padding_dep_matrix[i][:len(dep_matrix)] = dep_matrix[i]

            assert len(input_ids) == self.args.max_seq_len
            assert len(attention_mask) == self.args.max_seq_len
            assert len(token_type_ids) == self.args.max_seq_len
            assert len(valid) == self.args.max_seq_len
            assert len(padding_dep_matrix) == self.args.max_seq_len
            assert len(pos_ids) == self.args.max_seq_len
            assert len(dep_ids) == self.args.max_seq_len

            features.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "valid_ids": valid,
                "padding_dep_matrix": padding_dep_matrix,
                "relation": relation_id,
                "relative_subj": relative_subj,
                "relative_obj": relative_subj,
                "pos_ids": pos_ids,
                "dep_ids": dep_ids
            })
        return features

    def get_context_adj(self, tokens_len):
        adj = torch.zeros((tokens_len, tokens_len))
        for i in range(tokens_len):
            if i != tokens_len - 1:
                adj[i, i + 1] = 1
            if i != 0:
                adj[i - 1, i] = 1
        return adj

    def convert_to_glove_features(self):
        features = []
        vocab_save_path = os.path.join(self.args.vocab_dir, self.args.task_name, "vocab.pkl")
        emb_save_path = os.path.join(self.args.vocab_dir, self.args.task_name, "embedding.npy")
        if not os.path.exists(vocab_save_path) or not os.path.exists(emb_save_path):
            self.build_glove_embedding()
        dep_dict, rel_dict = self.get_deps_and_relations()
        pos_dict = self.get_POS()
        data_dir = os.path.join(self.args.data_dir, self.args.task_name, self.mode + ".json")

        with open(vocab_save_path, "rb") as v:
            vocabs = list(pickle.load(v))
        emb = np.load(emb_save_path)
        with open(data_dir, 'r') as f:
            items = json.load(f)
        items = self.process_data(items)
        max_seq_len = self.args.max_seq_len
        vocabs = {v: i for i, v in enumerate(vocabs)}
        for index, item in enumerate(items):
            subj_pos = [item["subj_start"], item["subj_end"]]
            obj_pos = [item["obj_start"], item["obj_end"]]
            relative_subj, relative_obj = self.compute_relative_distance(len(item["token"]), subj_pos, obj_pos)
            dep_matrix = self.generate_dep_matrix(item["stanford_head"])
            ctx_matrix = self.get_context_adj(len(item["token"]))
            pos_ids = [pos_dict[p] for p in item["stanford_pos"]]
            dep_ids = [dep_dict[d] for d in item["stanford_deprel"]]

            realtion_id = rel_dict[item["relation"]]
            input_ids = [vocabs[t] for t in item["token"]]
            input_ids += [vocabs["PAD_TOKEN"]] * (max_seq_len - len(input_ids))
            pos_ids += [pos_dict["none"]] * (max_seq_len - len(pos_ids))
            dep_ids += [dep_dict["none"]] * (max_seq_len - len(dep_ids))
            relative_subj += [127] * (max_seq_len - len(relative_subj))
            relative_obj += [127] * (max_seq_len - len(relative_obj))
            padding_dep_matrix = torch.zeros(max_seq_len, max_seq_len)
            padding_ctx_matrix = torch.zeros(max_seq_len, max_seq_len)
            for i in range(len(dep_matrix)):
                padding_dep_matrix[i][:len(dep_matrix)] = dep_matrix[i]
            for i in range(len(ctx_matrix)):
                padding_ctx_matrix[i][:len(ctx_matrix)] = ctx_matrix[i]
            input_emb = torch.from_numpy(emb[input_ids].astype(np.float32))

            assert len(input_ids) == max_seq_len
            assert len(padding_dep_matrix) == max_seq_len
            assert len(padding_ctx_matrix) == max_seq_len
            assert len(pos_ids) == max_seq_len
            assert len(dep_ids) == max_seq_len
            assert len(relative_subj) == max_seq_len
            assert len(relative_obj) == max_seq_len

            features.append({
                "input_ids": input_emb,
                "padding_dep_matrix": padding_dep_matrix,
                "relation": realtion_id,
                "relative_subj": relative_subj,
                "relative_obj": relative_obj,
                "pos_ids": pos_ids,
                "dep_ids": dep_ids,
                "padding_context_matrix": padding_ctx_matrix
            })
        return features


def load_tokens(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        processed_tokens = []
        for item in data:
            tokens = item['token']
            subj_pos = [item["subj_start"], item["subj_end"]]
            obj_pos = [item["obj_start"], item["obj_end"]]
            if "subj_type" in item:
                tokens[subj_pos[0]: subj_pos[1] + 1] = ["SUBJ-" + item["subj_type"]]
                tokens[obj_pos[0]: obj_pos[1] + 1] = ["OBJ-" + item["obj_type"]]
            else:
                tokens[subj_pos[0]: subj_pos[1] + 1] = ["SUBJ"]
                tokens[obj_pos[0]: obj_pos[1] + 1] = ["OBJ"]
            processed_tokens += tokens
        return processed_tokens


def split_train_dev(data_path, obj_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_set, dev_set = train_test_split(data, test_size=0.2, random_state=42)
    train_path = os.path.join(obj_path, "train.json")
    dev_path = os.path.join(obj_path, "dev.json")
    with open(train_path, "w", encoding="utf-8") as trainpath, open(dev_path, "w", encoding="utf-8") as devpath:
        json.dump(train_set, trainpath)
        json.dump(dev_set, devpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./datasets/')
    parser.add_argument('--task_name', type=str, default='semeval')
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--vocab_dir", type=str, default='./datasets/vocab/')
    parser.add_argument("--glove_dir", type=str, default='./glove/glove.840B.300d.txt')
    parser.add_argument("--glove_dim", type=int, default=300)
    parser.add_argument("--use_bert", type=bool, default=False)
    args = parser.parse_args()
    processor = REProcessor(args, "train")
    tokenizer = BertTokenizer.from_pretrained('../bert-base-cased')
    semeval_set = REDataset(args, processor, tokenizer)
    print(semeval_set[0])
    # split_train_dev("./datasets/train.json", "./datasets/semeval")
