import torch
from torch import nn
import torch.nn.functional as F
from models.GCN import GraphConvLayer, GCN, SelfAttention, MultiHeadAttnGCN, MultiHeadAttention


class DAGCN(nn.Module):
    def __init__(self, args):
        super(DAGCN, self).__init__()
        self.num_layers = args.num_layers
        self.num_att_layers = args.l_att
        self.pos1_emb = nn.Embedding(args.max_seq_len, args.relative_dim, padding_idx=args.relative_padding)
        self.pos2_emb = nn.Embedding(args.max_seq_len, args.relative_dim, padding_idx=args.relative_padding)
        self.pos_emb = nn.Embedding(args.num_pos, args.pos_dim, padding_idx=args.pos_padding)
        self.dep_emb = nn.Embedding(args.num_dep, args.dep_dim, padding_idx=args.dep_padding)

        self.relu = nn.LeakyReLU()
        self.total_dim = args.glove_dim + args.relative_dim * 2 + args.pos_dim + args.dep_dim  # total_dim=420
        ctx_dim = (self.total_dim - args.syn_dim)   # syn_dim=270, ctx_dim=150
        assert ctx_dim % 2 == 0
        self.context_encoder = nn.LSTM(self.total_dim, int(ctx_dim/2), bidirectional=True)
        self.dep_encoder = GCN(self.total_dim, args.syn_dim, drop_out=args.drop_out)
        # multiheadattn's parameters: q_dim, k_dim, v_dim, num_heads, hidden_size, dropout
        self.cxt2dep_attns = nn.ModuleList([
            MultiHeadAttention(ctx_dim, args.syn_dim, args.syn_dim, args.num_heads, args.syn_dim, args.drop_out)
            for _ in range(self.num_att_layers)
        ])
        self.dep2ctx_attns = nn.ModuleList([
            MultiHeadAttention(args.syn_dim, ctx_dim, ctx_dim, args.num_heads, ctx_dim, args.drop_out)
            for _ in range(self.num_att_layers)
             ])

        self.W_c = nn.Linear((args.num_layers + 1) * self.total_dim, self.total_dim)
        self.W_d = nn.Linear((args.num_layers + 1) * self.total_dim, self.total_dim)
        self.W_g = nn.Linear(self.total_dim * 2, self.total_dim * 2)
        self.classifier = nn.Linear(self.total_dim * 6, args.num_relations)

    def forward(self, input_ids, relation_id, dep_adj, relative_subj, relative_obj, pos_ids, dep_ids, ctx_adj):
        pos_emb = self.pos_emb(pos_ids)  # 30
        dep_emb = self.dep_emb(dep_ids)  # 30
        sub_pos_emb = self.pos1_emb(relative_subj)  # 30
        obj_pos_emb = self.pos2_emb(relative_obj)  # 30
        e_emb = torch.cat((input_ids, pos_emb, dep_emb, sub_pos_emb, obj_pos_emb), dim=-1)  # 420
        ctx_init_emb = self.context_encoder(e_emb)[0]  # 150
        dep_init_emb = self.dep_encoder(dep_adj, e_emb)  # 270
        e_init = torch.cat((ctx_init_emb, dep_init_emb), dim=-1)  # 420 = total_dim
        assert e_init.size(-1) == self.total_dim
        h_c = e_init
        h_d = e_init
        h_c_all = [e_init]
        h_d_all = [e_init]
        for layer in range(self.num_layers):
            h_c_ = self.context_encoder(h_c)[0]  # 150
            h_d_ = self.dep_encoder(dep_adj, h_d)  # 270
            h_c2d_temp = h_d_
            h_d2c_temp = h_c_
            for index in range(self.num_att_layers):
                h_c2d_temp = self.cxt2dep_attns[index](h_c_, h_c2d_temp, h_c2d_temp, dep_adj)  # 270
                h_d2c_temp = self.dep2ctx_attns[index](h_d_, h_d2c_temp, h_d2c_temp, ctx_adj)  # 150
            h_c2d = h_d_ + h_c2d_temp  # 270
            h_d2c = h_c_ + h_d2c_temp  # 150
            h_c = torch.concat([h_c_, h_c2d], dim=-1)  # 420
            h_d = torch.concat([h_d_, h_d2c], dim=-1)  # 420
            h_c_all.append(h_c)
            h_d_all.append(h_d)

        # aggregate
        h_c_all = self.W_c(torch.concat(h_c_all, dim=-1))  # (num_layers+1) * total_dim -> total_dim
        h_d_all = self.W_d(torch.concat(h_d_all, dim=-1))  # (num_layers+1) * total_dim -> total_dim
        # gate
        h_all = torch.concat([h_c_all, h_d_all], dim=-1)
        g = F.sigmoid(self.W_g(h_all))
        h_agg = g * h_all

        h_sen = torch.max(h_agg, dim=1).values  # total_dim*2
        mask_subj = relative_subj == 0
        mask_subj = mask_subj.unsqueeze(-1).expand(-1, -1, h_agg.size(-1))
        mask_obj = relative_obj == 0
        mask_obj = mask_obj.unsqueeze(-1).expand(-1, -1, h_agg.size(-1))
        h_subj = torch.max(h_agg.masked_fill(~mask_subj, 0), dim=1).values  # total_dim * 2
        h_obj = torch.max(h_agg.masked_fill(~mask_obj, 0), dim=1).values  # total_dim  * 2
        h_final = torch.concat([h_sen, h_subj, h_obj], dim=-1)

        logits = self.classifier(h_final)

        if relation_id is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits, relation_id)
            return logits, loss
        return logits, "none"
