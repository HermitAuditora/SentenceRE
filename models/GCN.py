from torch import nn
import torch.nn.functional as F
import torch
import math





class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, drop_out=0.5):
        super(GCN, self).__init__()
        self.transform = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, adj, features):
        denom = adj.sum(dim=-1, keepdim=True) + 1
        agg = torch.matmul(adj, features)
        self_loop = features + agg
        normalization = self_loop / denom
        transform = self.transform(normalization)
        transform = self.dropout(transform)
        return transform

class MultiHeadAttnGCN(nn.Module):
    def __init__(self, hidden_dim, num_heads, drop_out=0.5):
        super(MultiHeadAttnGCN, self).__init__()
        assert hidden_dim % num_heads == 0
        self.head_dim = int(hidden_dim / num_heads)
        self.multi_gcns = nn.ModuleList()
        for i in range(num_heads):
            self.multi_gcns.append(nn.Linear(hidden_dim, self.head_dim))
            self.multi_gcns.append(GCN(self.head_dim, drop_out))
            self.multi_gcns.append(SelfAttention(self.head_dim))
            self.multi_gcns.append(GCN(self.head_dim, drop_out))
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, adj, features):
        temp_out = []
        for i in range(0, len(self.multi_gcns), 4):
            transform = self.multi_gcns[i](features)
            gcn_out = self.multi_gcns[i+1](adj, transform)
            attn_score = self.multi_gcns[i+2](adj, gcn_out)
            gcn_out = self.multi_gcns[i+3](attn_score, gcn_out)
            temp_out.append(gcn_out)
        output = torch.cat(temp_out, dim=-1)
        output = self.linear_out(output)
        return output
class GraphConvLayer(nn.Module):

    def __init__(self, hidden_dim, num_heads, drop_out=0.5):
        super(GraphConvLayer, self).__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(drop_out)
        self.transform = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.num_heads):
            self.weight_list.append(nn.Linear(self.hidden_dim + self.head_dim * i, self.head_dim))

    def forward(self, adj, features):
        # add 1 to prevent denominator being 0
        denom = adj.sum(dim=2).unsqueeze(2) + 1

        output = features
        cache = [output]
        outputs = []
        for i in range(self.num_heads):
            agg = torch.matmul(adj, output)
            transform = self.weight_list[i](agg)
            self_loop = transform + self.weight_list[i](output)
            normalization = self_loop / denom
            normalization = F.leaky_relu(normalization)
            cache.append(normalization)
            output = torch.concat(cache, dim=-1)
            outputs.append(normalization)
        gcn_output = torch.cat(outputs, dim=-1)
        resduial = gcn_output + features

        last_transform = self.transform(resduial)
        return last_transform


class MultiHeadAttention(nn.Module):

    def __init__(self, q_dim, k_dim, v_dim, num_heads, hidden_size, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, "hidden_size:{}, num_head:{}".format(hidden_size, num_heads)
        self.head_dim = hidden_size // num_heads
        self.W_k = nn.Linear(q_dim, hidden_size)
        self.W_v = nn.Linear(k_dim, hidden_size)
        self.W_v = nn.Linear(v_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        transform_q = self.W_k(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        transform_k = self.W_v(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        transform_v = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(transform_q, transform_k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, float(-1e9))
        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores, transform_v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return output



class SelfAttention(nn.Module):
    def __init__(self, features_dim):
        super(SelfAttention, self).__init__()
        self.query_linear = nn.Linear(features_dim, features_dim)
        self.key_linear = nn.Linear(features_dim, features_dim)

    def forward(self, dep_adj, features):
        query = self.query_linear(features)
        key = self.key_linear(features)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(features.size(-1))
        scores = scores.masked_fill(dep_adj == 0, -1e9)
        weight_scores = torch.softmax(scores, dim=-1)
        return weight_scores




