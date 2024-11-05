import argparse


class BaseConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_dir', type=str, default='./data/datasets/',
                                 help='base data directory')
        self.parser.add_argument('--task_name', type=str, default='semeval', help='name of the task')
        self.parser.add_argument("--max_seq_len", type=int, default=128, help='maximum total input sequence length '
                                                                              'after tokenization for bert')
        self.parser.add_argument("--vocab_dir", type=str, default='./data/datasets/vocab/',
                                 help='directory to save vocabulary for glove')
        self.parser.add_argument("--glove_dir", type=str, default='./data/glove/glove.840B.300d.txt',
                                 help='directory to load glove embeddings')
        self.parser.add_argument("--glove_dim", type=int, default=300, help='dimension of glove embeddings')
        self.parser.add_argument("--use_bert", type=bool, default=False, help='whether to use BERT pre-trained model')
        self.parser.add_argument("--batch_size", type=int, default=32, help='batch size')
        self.parser.add_argument("--num_workers", type=float, default=4, help='number of workers')
        self.parser.add_argument("--learning_rate", type=float, default=5e-4, help='learning rate')
        self.parser.add_argument("--eps", type=float, default=1e-8, help='epsilon for Adam')
        self.parser.add_argument("--output_dir", type=str, default="./output/", help='output directory')
        self.parser.add_argument("--seed", type=int, default=1234, help='seed for initializing training. ')
        self.parser.add_argument("--num_gcns", type=int, default=2, help='number of gcns')
        self.parser.add_argument("--dropout", type=float, default=0.5, help='dropout rate')
        self.parser.add_argument("--num_heads", type=int, default=6, help='number of attention heads')
        self.parser.add_argument("--hidden_size", type=int, default=300, help='dimension of hidden state')
        self.parser.add_argument("--dep_dim", type=int, default=30, help='dimension of deprel')
        self.parser.add_argument("--pos_dim", type=int, default=30, help='dimension of pos')
        self.parser.add_argument("--relative_dim", type=int, default=30, help='dimension of relative distance from entities')
        self.parser.add_argument("--drop_out", type=float, default=0.5, help='dropout rate')
        self.parser.add_argument("--num_layers", type=int, default=3, help='number of interaction layers')
        self.parser.add_argument("--syn_dim", type=int, default=270, help='output dimension of dep encoder')
        self.parser.add_argument("--l_att", type=int, default=2, help='number of attention layers')

        # -----------------------train_config----------------------------------------------
        self.parser.add_argument("--num_epochs", type=int, default=200, help='number of epochs during training')
        self.parser.add_argument("--early_stop", type=int, default=200, help='early stopping patience')

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args
