import os

import torch
import tqdm
from torch.optim import AdamW, SGD
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import random
from configs.baseconfig import BaseConfig
from data.preprocess import *
from metrics import Metrics
from main_model import DAGCN


def save_models(output_dir, model):
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), output_model_file)


def train(args, train_set, dev_set, model, device):
    global_step = 0
    train_loss = 0
    best_f1 = 0

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.eps)
    # optimizer = SGD(model.parameters(), lr=args.learning_rate)

    print("lr: {} eps:{}".format(args.learning_rate, args.eps))
    num_relations = train_set.get_num_relations()
    early_stop = 0
    for epoch in range(args.num_epochs):
        model.train()
        metrics = Metrics(num_relations)
        train_iter = tqdm.tqdm(train_loader, desc="Epoch {}/{}".format(epoch + 1, args.num_epochs))
        for step, batch in enumerate(train_iter):
            sub_metrics = Metrics(num_relations)
            batch = tuple(t.to(device) for t in batch)
            if args.use_bert:
                input_ids, attention_mask, token_type_ids, valid_ids, relation_id, dep_adj, relative_subj, relative_obj, pos_ids, dep_ids, ctx_adj = batch
                logits, loss = model(input_ids, attention_mask, token_type_ids, valid_ids, relation_id, dep_adj,
                                     relative_subj, relative_obj, pos_ids, dep_ids, ctx_adj)
            else:
                input_ids, relation_id, dep_adj, relative_subj, relative_obj, pos_ids, dep_ids, ctx_adj = batch
                logits, loss = model(input_ids, relation_id, dep_adj, relative_subj, relative_obj, pos_ids, dep_ids, ctx_adj)
            pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            sub_metrics.add(pred, relation_id)
            metrics.add(pred, relation_id)
            loss.backward()
            train_iter.update(1)

            train_iter.set_postfix_str(
                "Step:{} Loss:{:.4f}".format(global_step, loss.item())
            )
            global_step += 1
            optimizer.step()
            optimizer.zero_grad()

        eval_precision, eval_recall, eval_f1 = eval(args, model, dev_set, device)
        if eval_f1 > best_f1:
            early_stop = 0
            keep_precision = eval_precision
            keep_recall = eval_recall
            best_f1 = eval_f1
            save_path = os.path.join(args.output_dir, args.task_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_models(save_path, model)
        if early_stop == args.early_stop:
            print("Early stopping at epoch {}".format(epoch+1))
            break
        early_stop += 1

    print("best precision: {:.4f}\nbest recall: {:.4f}\nbest f1-score: {:.4f}".format(keep_precision, keep_recall, best_f1))

def eval(args, model, test_set, device):
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model.eval()
    num_relations = test_set.get_num_relations()
    metrics = Metrics(num_relations)
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            if args.use_bert:
                input_ids, attention_mask, token_type_ids, valid_ids, relation_id, dep_adj, relative_subj, relative_obj, pos_ids, dep_ids, ctx_adj = batch
                logits, loss = model(input_ids, attention_mask, token_type_ids, valid_ids, relation_id, dep_adj,
                                     relative_subj, relative_obj, pos_ids, dep_ids, ctx_adj)
            else:
                input_ids, relation_id, dep_adj, relative_subj, relative_obj, pos_ids, dep_ids, ctx_adj = batch
                logits, loss = model(input_ids, relation_id, dep_adj, relative_subj, relative_obj, pos_ids, dep_ids, ctx_adj)
            pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            metrics.add(pred, relation_id)
    print("Evaluation metrics: \n macro_precision: {:.4f} \n macro_recall: {:.4f}\n macro_f1-score: {:.4f}".format(
        metrics.get_precision("macro"),
        metrics.get_recall("macro"),
        metrics.get_f1("macro")
    ))
    return metrics.get_precision("macro"), metrics.get_recall("macro"), metrics.get_f1("macro")


def main():
    config = BaseConfig()
    args = config.parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # train process
    tokenizer = BertTokenizer.from_pretrained('./bert-base-cased')
    train_processor = REProcessor(args, "train")
    dev_processor = REProcessor(args, "dev")
    train_set = REDataset(args, train_processor, tokenizer)
    dev_set = REDataset(args, dev_processor, tokenizer)
    pos_dict = train_processor.get_POS()
    dep_dict, rel_dict = train_processor.get_deps_and_relations()
    num_pos, num_dep, num_relations = train_processor.get_pos_dep_rel_len()
    args.__dict__["num_pos"] = num_pos
    args.__dict__["num_dep"] = num_dep
    args.__dict__["num_relations"] = num_relations
    args.__dict__["relative_padding"] = 127
    args.__dict__["pos_padding"] = pos_dict["none"]
    args.__dict__["dep_padding"] = dep_dict["none"]
    model = DAGCN(args)
    model = model.to(device)
    train(args, train_set, dev_set, model, device)

    print("Testing")
    # test process
    test_processer = REProcessor(args, "test")
    test_set = REDataset(args, test_processer, tokenizer)
    model_path = args.output_dir + args.task_name + "/pytorch_model.bin"
    model.load_state_dict(torch.load(model_path))
    eval(args, model, test_set, device)


if __name__ == "__main__":
    main()
