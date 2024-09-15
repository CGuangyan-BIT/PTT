from lib.pointops2.functions import pointops
import torch

query = torch.arange(24).reshape(8,1,3).cuda()
key = query.clone()
index_kv = torch.arange(8).cuda()
index_query_offsets = torch.arange(8).cuda()
n_max = 2

attn_flat = pointops.attention_step1_v2(query.float(), key.float(), index_kv.int(), index_query_offsets.int(), n_max)

print(attn_flat)