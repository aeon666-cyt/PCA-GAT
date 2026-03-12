import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import time
import os
import json
from collections import defaultdict

from config_pcagat import *
from dataset_machining import MachiningDataset
from utils_pcagat import evaluate_model, compute_constraint_satisfaction

SEED = 42
DIM = 64
EVAL_K = [1, 3, 5]

SPARSITY_RATIOS = [0.8, 0.6, 0.4]

METHOD_NAMES = [
    "BPR-MF", "LightGCN", "CKE", "TransE-Rec",
    "KGAT", "KGIN", "CB-KG",
]


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_dataset_with_ratio(ratio):
    original_subsample = MachiningDataset.subsample_train
    original_inject = MachiningDataset.inject_noise

    MachiningDataset.subsample_train = lambda self, ratio=0.6: None
    MachiningDataset.inject_noise = lambda self, noise_ratio=0.1: None

    try:
        dataset = MachiningDataset(DATASET_PATH, device=str(DEVICE))
    finally:
        MachiningDataset.subsample_train = original_subsample
        MachiningDataset.inject_noise = original_inject

    if ratio < 1.0:
        dataset.subsample_train(ratio=ratio)

    dataset.user_hard_negs = {}
    dataset._build_hard_neg_pools()

    return dataset


def compute_metrics_manual(ranked_list, ground_truth, k):
    hits, dcg = 0, 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            hits += 1
            dcg += 1.0 / np.log2(i + 2)
    recall = hits / len(ground_truth) if ground_truth else 0
    hr = 1.0 if hits > 0 else 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    ndcg = dcg / idcg if idcg > 0 else 0
    return recall, hr, ndcg, 0.0


def evaluate_scores(all_scores, dataset, top_k_list=EVAL_K):
    results = {k: {'Recall': [], 'HR': [], 'NDCG': [], 'MRR': [], 'Precision': []}
               for k in top_k_list}

    users = list(dataset.test_dict.keys())

    for u in users:
        if isinstance(all_scores, dict):
            scores = all_scores.get(u)
            if scores is None:
                continue
            scores = np.array(scores).copy()
        else:
            scores = all_scores[u].copy()

        for i in dataset.train_set.get(u, []):
            scores[i] = -1e9

        ranked = np.argsort(scores)[::-1]
        gt = set(dataset.test_dict[u])
        if not gt:
            continue

        for k in top_k_list:
            recall, hr, ndcg, mrr = compute_metrics_manual(ranked[:k], gt, k)
            results[k]['Recall'].append(recall)
            results[k]['HR'].append(hr)
            results[k]['NDCG'].append(ndcg)
            results[k]['MRR'].append(mrr)
            results[k]['Precision'].append(recall)

    avg = {}
    for k in top_k_list:
        avg[k] = {
            'Recall': np.mean(results[k]['Recall']) if results[k]['Recall'] else 0,
            'HR': np.mean(results[k]['HR']) if results[k]['HR'] else 0,
            'NDCG': np.mean(results[k]['NDCG']) if results[k]['NDCG'] else 0,
            'MRR': np.mean(results[k]['MRR']) if results[k]['MRR'] else 0,
            'Precision': np.mean(results[k]['Precision']) if results[k]['Precision'] else 0,
        }
    return avg


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, u, i, neg_i):
        u_e = self.user_emb(u)
        return (torch.sum(u_e * self.item_emb(i), dim=1),
                torch.sum(u_e * self.item_emb(neg_i), dim=1))


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim, graph, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.graph = graph
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def computer(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight]).float()
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph.float(), all_emb)
            embs.append(all_emb)
        final = torch.mean(torch.stack(embs, 1), 1)
        return torch.split(final, [self.n_users, self.n_items])

    def forward(self, u, i, neg_i):
        u_all, i_all = self.computer()
        return (torch.sum(u_all[u] * i_all[i], dim=1),
                torch.sum(u_all[u] * i_all[neg_i], dim=1))


def build_lightgcn_graph(dataset):
    nu, ni = dataset.n_users, dataset.n_items
    n = nu + ni
    rows, cols = [], []
    for u, items in dataset.train_set.items():
        for i in items:
            rows.append(u); cols.append(nu + i)
            rows.append(nu + i); cols.append(u)
    vals = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
    rs = np.array(adj.sum(1)).flatten()
    d = np.power(rs, -0.5); d[np.isinf(d)] = 0.0
    norm = sp.diags(d).dot(adj).dot(sp.diags(d)).tocoo()
    idx = torch.LongTensor(np.vstack([norm.row, norm.col]))
    val = torch.FloatTensor(norm.data)
    return torch.sparse_coo_tensor(idx, val, (n, n)).to(DEVICE)


class CKE(nn.Module):
    def __init__(self, n_users, n_items, n_entities, n_relations,
                 dim, item2entity, device=DEVICE):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.dim = dim
        self.device = device
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb_cf = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb_cf.weight, std=0.1)
        self.entity_emb = nn.Embedding(n_entities, dim)
        self.relation_emb = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.register_buffer('item2entity_map',
                             torch.tensor([item2entity.get(i, 0) for i in range(n_items)],
                                          dtype=torch.long))

    def get_item_embeddings(self):
        return self.item_emb_cf.weight + self.entity_emb(self.item2entity_map)

    def forward(self, u, i, neg_i):
        u_e = self.user_emb(u)
        ie = self.get_item_embeddings()
        return (torch.sum(u_e * ie[i], 1), torch.sum(u_e * ie[neg_i], 1))

    def kg_loss(self, h, r, t, neg_t):
        he, re = self.entity_emb(h), self.relation_emb(r)
        te, ne = self.entity_emb(t), self.entity_emb(neg_t)
        return torch.mean(F.relu(
            torch.norm(he + re - te, 1, -1) -
            torch.norm(he + re - ne, 1, -1) + 1.0))


class TransERec(nn.Module):
    def __init__(self, n_entities, n_relations, dim, device=DEVICE):
        super().__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        self.device = device
        self.entity_emb = nn.Embedding(n_entities, dim)
        self.relation_emb = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t, neg_t):
        he = self.entity_emb(h)
        re = self.relation_emb(r)
        te = self.entity_emb(t)
        ne = self.entity_emb(neg_t)
        pos = torch.norm(he + re - te, p=1, dim=1)
        neg = torch.norm(he + re - ne, p=1, dim=1)
        return torch.mean(F.relu(pos - neg + 1.0))


def train_transe_rec(dataset, dim=DIM, epochs=200, lr=0.001):
    set_seed()
    model = TransERec(dataset.n_entities, dataset.n_relations,
                      dim, DEVICE).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    kg_loader = dataset.get_kg_loader(batch_size=1024)

    best_loss = float('inf')
    patience_cnt = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in kg_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss = model(batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(kg_loader)

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= 20 and ep > 50:
            break

    model.eval()
    with torch.no_grad():
        entity_embs = model.entity_emb.weight.cpu().numpy()

    item_embs = np.zeros((dataset.n_items, dim), dtype=np.float32)
    for i in range(dataset.n_items):
        eid = dataset.item2entity.get(i, 0)
        if eid < dataset.n_entities:
            item_embs[i] = entity_embs[eid]

    all_scores = np.zeros((dataset.n_users, dataset.n_items), dtype=np.float32)

    for u in range(dataset.n_users):
        train_items = list(dataset.train_set.get(u, []))
        if len(train_items) == 0:
            continue
        user_ent_ids = [dataset.item2entity.get(i, 0) for i in train_items]
        user_repr = np.mean([entity_embs[eid] for eid in user_ent_ids
                             if eid < dataset.n_entities], axis=0)
        if user_repr is None or np.isnan(user_repr).any():
            continue
        user_norm = np.linalg.norm(user_repr)
        item_norms = np.linalg.norm(item_embs, axis=1)
        if user_norm > 0:
            cos_sim = item_embs @ user_repr / (item_norms * user_norm + 1e-10)
            all_scores[u] = cos_sim

    return all_scores


class KGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_relations, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W_r = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(3 * out_dim, 1, bias=False)
        self.relation_emb = nn.Embedding(n_relations, in_dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_type):
        src, dst = edge_index[0], edge_index[1]
        xt = self.W(x)
        r = self.W_r(self.relation_emb(edge_type))
        att = self.leaky_relu(
            self.a(torch.cat([xt[src], r, xt[dst]], -1))).squeeze(-1)
        att = att - att.max()
        att_exp = torch.exp(att)
        N = x.size(0)
        att_sum = torch.zeros(N, device=x.device)
        att_sum.scatter_add_(0, dst, att_exp)
        att_norm = self.dropout(att_exp / (att_sum[dst] + 1e-10))
        msg = att_norm.unsqueeze(-1) * xt[src]
        out = torch.zeros(N, self.out_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return out


class KGAT(nn.Module):
    def __init__(self, n_users, n_entities, n_relations, n_items,
                 dim, item2entity, edge_index, edge_type,
                 n_layers=1, dropout=0.1, device='cpu'):
        super().__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.device = device
        self.register_buffer('item2entity_map',
                             torch.tensor([item2entity.get(i, 0)
                                           for i in range(n_items)],
                                          dtype=torch.long))
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_type', edge_type)
        self.node_emb = nn.Embedding(n_users + n_entities, dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        self.gat_layers = nn.ModuleList([
            KGATLayer(dim, dim, n_relations, dropout)
            for _ in range(n_layers)
        ])
        self.kg_relation_emb = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.kg_relation_emb.weight)

    def forward(self):
        x = self.node_emb.weight
        all_emb = [x]
        for layer in self.gat_layers:
            x = F.normalize(
                F.elu(layer(x, self.edge_index, self.edge_type)),
                p=2, dim=-1)
            all_emb.append(x)
        final = torch.mean(torch.stack(all_emb, 0), 0)
        return final[:self.n_users], final[self.n_users:]

    def get_item_embeddings(self, entity_emb):
        return entity_emb[self.item2entity_map]

    def bpr_loss(self, users, pos, neg):
        ue, ee = self.forward()
        ie = self.get_item_embeddings(ee)
        return -torch.mean(F.logsigmoid(
            (ue[users] * ie[pos]).sum(-1) -
            (ue[users] * ie[neg]).sum(-1)))

    def kg_loss(self):
        ego = self.node_emb.weight
        interact_rel = self.edge_type.max().item()
        mask = ((self.edge_index[0] >= self.n_users) &
                (self.edge_type != interact_rel))
        src = self.edge_index[0][mask]
        dst = self.edge_index[1][mask]
        rel = self.edge_type[mask]
        if src.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        n = min(1024, src.size(0))
        idx = torch.randint(0, src.size(0), (n,), device=self.device)
        h = ego[src[idx]]
        t = ego[dst[idx]]
        r = self.kg_relation_emb(rel[idx])
        neg_t = ego[self.n_users + torch.randint(
            0, self.n_entities, (n,), device=self.device)]
        return torch.mean(F.relu(
            torch.norm(h + r - t, 2, -1) -
            torch.norm(h + r - neg_t, 2, -1) + 1.0))


class KGIN(nn.Module):
    def __init__(self, n_users, n_items, n_entities, n_relations,
                 dim, item2entity, train_set,
                 kg_edge_index, kg_edge_type,
                 n_intents=2, n_layers=1, dropout=0.1, device=DEVICE):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        self.n_intents = n_intents
        self.n_layers = n_layers
        self.device = device

        self.user_emb = nn.Embedding(n_users, dim)
        self.entity_emb = nn.Embedding(n_entities, dim)

        self.n_relations_total = n_relations * 2
        self.relation_emb = nn.Embedding(self.n_relations_total, dim)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.register_buffer(
            'item2entity_map',
            torch.tensor([item2entity.get(i, 0) for i in range(n_items)],
                         dtype=torch.long))
        self.register_buffer('kg_edge_index', kg_edge_index)
        self.register_buffer('kg_edge_type', kg_edge_type)

        self._build_ui_graph(train_set)

        self.intent_weights = nn.Parameter(
            torch.randn(n_intents, self.n_relations_total))
        nn.init.xavier_uniform_(self.intent_weights)
        self.intent_router = nn.Linear(dim, n_intents)

        self.kg_layers = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def _build_ui_graph(self, train_set):
        n_nodes = self.n_users + self.n_items
        rows, cols = [], []
        for u, items in train_set.items():
            for i in items:
                rows.append(u)
                cols.append(self.n_users + i)
                rows.append(self.n_users + i)
                cols.append(u)
        if not rows:
            self.register_buffer('ui_graph', torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0), (n_nodes, n_nodes)))
            return
        vals = np.ones(len(rows), dtype=np.float32)
        adj = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
        rs = np.array(adj.sum(1)).flatten()
        d = np.power(rs, -0.5)
        d[np.isinf(d)] = 0.0
        norm = sp.diags(d).dot(adj).dot(sp.diags(d)).tocoo()
        idx = torch.LongTensor(np.vstack([norm.row, norm.col]))
        val = torch.FloatTensor(norm.data)
        self.register_buffer(
            'ui_graph',
            torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes)))

    def _propagate_kg(self):
        x = self.entity_emb.weight
        src = self.kg_edge_index[0]
        dst = self.kg_edge_index[1]
        rel = self.kg_edge_type

        for layer in self.kg_layers:
            if src.size(0) == 0:
                x = F.elu(layer(x))
                continue

            r = self.relation_emb(rel)

            msg = layer(x[src]) * torch.sigmoid(r)
            agg = torch.zeros_like(x)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
            deg = torch.zeros(self.n_entities, device=self.device)
            deg.scatter_add_(0, dst,
                             torch.ones_like(dst, dtype=torch.float))
            x = F.normalize(
                F.elu(agg / deg.clamp(min=1).unsqueeze(-1) + x),
                p=2, dim=-1)
            x = self.dropout(x)

        return x

    def _propagate_ui(self, user_emb, item_emb):
        all_emb = torch.cat([user_emb, item_emb], 0).float()
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.ui_graph.float(), all_emb)
            embs.append(all_emb)
        final = torch.mean(torch.stack(embs, 1), 1)
        return final[:self.n_users], final[self.n_users:]

    def get_full_embeddings(self):
        kg_ent = self._propagate_kg()
        item_kg = kg_ent[self.item2entity_map]
        u_prop, i_prop = self._propagate_ui(
            self.user_emb.weight, item_kg)

        intent_dist = F.softmax(self.intent_router(u_prop), -1)
        intent_emb = torch.matmul(
            F.softmax(self.intent_weights, -1),
            self.relation_emb.weight)
        user_intent = torch.matmul(intent_dist, intent_emb)

        return u_prop + user_intent, i_prop + item_kg

    def forward(self, u, i, neg_i):
        ue, ie = self.get_full_embeddings()
        return (torch.sum(ue[u] * ie[i], 1),
                torch.sum(ue[u] * ie[neg_i], 1))

    def kg_loss(self, h, r, t, neg_t):
        he = self.entity_emb(h)
        re = self.relation_emb(r)
        te = self.entity_emb(t)
        ne = self.entity_emb(neg_t)
        return torch.mean(F.relu(
            torch.norm(he + re - te, 1, -1) -
            torch.norm(he + re - ne, 1, -1) + 1.0))


def _build_kg_edges_entity_space(dataset):
    kg_data = dataset.kg_data
    nr = dataset.n_relations
    ne = dataset.n_entities
    s, d, r = [], [], []
    for h, rel, t in kg_data:
        if h < ne and t < ne:
            s.append(h); d.append(t); r.append(rel)
            s.append(t); d.append(h); r.append(rel + nr)
    return torch.LongTensor([s, d]), torch.LongTensor(r)


def compute_cbkg_scores(dataset):
    n_users = dataset.n_users
    n_items = dataset.n_items

    item_features = {}
    for item_id in range(n_items):
        ent_id = dataset.item2entity.get(item_id, None)
        if ent_id is None:
            item_features[item_id] = set()
            continue
        neighbors = {ent_id}
        for r, t in dataset.kg_dict.get(ent_id, []):
            if t < dataset.n_entities:
                neighbors.add(t)
        item_features[item_id] = neighbors

    user_features = {}
    for u in range(n_users):
        feats = set()
        for item_id in dataset.train_set.get(u, []):
            feats.update(item_features.get(item_id, set()))
        user_features[u] = feats

    all_scores = np.zeros((n_users, n_items), dtype=np.float32)
    for u in range(n_users):
        u_feats = user_features[u]
        if not u_feats:
            continue
        for i in range(n_items):
            i_feats = item_features.get(i, set())
            if not i_feats:
                continue
            intersection = len(u_feats & i_feats)
            union = len(u_feats | i_feats)
            if union > 0:
                all_scores[u, i] = intersection / union

    return all_scores


def _train_loop(model, dataset, optimizer, epochs=1000, patience=30,
                eval_interval=5, min_epochs=30,
                kg_loader=None, kg_weight=0.1, model_name="Model"):
    loader = dataset.get_interaction_loader()
    best_r1, patience_cnt, best_state = 0, 0, None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        kg_iter = iter(kg_loader) if kg_loader else None

        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            if model_name == "KGAT":
                bpr = model.bpr_loss(batch[:, 0], batch[:, 1], batch[:, 2])
            else:
                pos, neg = model(batch[:, 0], batch[:, 1], batch[:, 2])
                bpr = -torch.mean(F.logsigmoid(pos - neg))

            kg_loss_val = torch.tensor(0.0, device=DEVICE)
            if kg_iter is not None and kg_weight > 0:
                try:
                    kb = next(kg_iter)
                except StopIteration:
                    kg_iter = iter(kg_loader)
                    kb = next(kg_iter)
                kb = kb.to(DEVICE)
                if model_name == "KGAT":
                    kg_loss_val = model.kg_loss()
                else:
                    kg_loss_val = model.kg_loss(
                        kb[:, 0], kb[:, 1], kb[:, 2], kb[:, 3])

            loss = bpr + kg_weight * kg_loss_val
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

        if ep >= min_epochs and ep % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                m = evaluate_model(model, dataset, top_k_list=[1, 5])
                r1 = m[1]['Recall']
                r5 = m[5]['Recall']

            if r1 > best_r1 + 1e-4:
                best_r1 = r1
                patience_cnt = 0
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1

            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_and_eval(method_name, dataset, edge_index, edge_type):
    set_seed()

    if method_name == "CB-KG":
        scores = compute_cbkg_scores(dataset)
        return evaluate_scores(scores, dataset, EVAL_K)

    if method_name == "TransE-Rec":
        scores = train_transe_rec(dataset, dim=DIM)
        return evaluate_scores(scores, dataset, EVAL_K)

    if method_name == "BPR-MF":
        model = BPRMF(dataset.n_users, dataset.n_items, DIM).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        model = _train_loop(model, dataset, opt, model_name="BPR-MF")

    elif method_name == "LightGCN":
        graph = build_lightgcn_graph(dataset)
        model = LightGCN(dataset.n_users, dataset.n_items,
                          DIM, graph, 3).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        model = _train_loop(model, dataset, opt, model_name="LightGCN")

    elif method_name == "CKE":
        model = CKE(dataset.n_users, dataset.n_items, dataset.n_entities,
                     dataset.n_relations, DIM, dataset.item2entity,
                     DEVICE).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        kg_loader = dataset.get_kg_loader()
        model = _train_loop(model, dataset, opt, kg_loader=kg_loader,
                            kg_weight=0.1, model_name="CKE")

    elif method_name == "KGAT":
        model = KGAT(
            dataset.n_users, dataset.n_entities,
            dataset.n_relations * 2 + 1, dataset.n_items,
            DIM, dataset.item2entity,
            edge_index.to(DEVICE), edge_type.to(DEVICE),
            n_layers=1, dropout=0.1, device=DEVICE
        ).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        kg_loader = dataset.get_kg_loader()
        model = _train_loop(model, dataset, opt, kg_loader=kg_loader,
                            kg_weight=0.1, model_name="KGAT")

    elif method_name == "KGIN":
        kg_ei, kg_et = _build_kg_edges_entity_space(dataset)
        model = KGIN(
            dataset.n_users, dataset.n_items, dataset.n_entities,
            dataset.n_relations, DIM, dataset.item2entity,
            dataset.train_set,
            kg_ei.to(DEVICE), kg_et.to(DEVICE),
            n_intents=2,
            n_layers=1,
            dropout=0.1, device=DEVICE
        ).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        kg_loader = dataset.get_kg_loader()
        model = _train_loop(model, dataset, opt, kg_loader=kg_loader,
                            kg_weight=0.1, model_name="KGIN")
    else:
        raise ValueError(f"Unknown method: {method_name}")

    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(model, dataset, top_k_list=EVAL_K)

    del model
    torch.cuda.empty_cache()
    return metrics


def main():
    t0 = time.time()
    all_results = {}

    for ratio in SPARSITY_RATIOS:
        dataset = load_dataset_with_ratio(ratio)
        edge_index, edge_type = dataset.get_edge_data()

        ratio_results = {}

        for method in METHOD_NAMES:
            try:
                metrics = train_and_eval(method, dataset,
                                         edge_index, edge_type)
                ratio_results[method] = metrics
            except Exception as e:
                import traceback
                traceback.print_exc()
                ratio_results[method] = None

        all_results[ratio] = ratio_results

        del dataset, edge_index, edge_type
        torch.cuda.empty_cache()

    total_time = time.time() - t0

    print("\n📊 Quick View: Recall@1")
    hdr = ("| Method       |" +
           "|".join(f" R@1({r:.0%}) " for r in SPARSITY_RATIOS) + "|")
    sep = "|---" * (len(SPARSITY_RATIOS) + 1) + "|"
    print(hdr)
    print(sep)
    for method in METHOD_NAMES:
        vals = []
        for ratio in SPARSITY_RATIOS:
            m = all_results[ratio].get(method)
            vals.append(f"{m[1]['Recall']:.4f}" if m else "FAIL")
        print(f"| {method:<12s} | " +
              " | ".join(f"{v:>8s}" for v in vals) + " |")

    print("\n\n📊 Full Table (for paper)")
    print("| Training | Method | Recall@1 | Recall@3 | Recall@5 | "
          "HR@5 | NDCG@5 | MRR@5 |")
    print("|---|---|---|---|---|---|---|---|")

    for ratio in SPARSITY_RATIOS:
        first_in_group = True
        for method in METHOD_NAMES:
            m = all_results[ratio].get(method)
            if m is None:
                print(f"| {ratio:.0%} | {method} | "
                      f"- | - | - | - | - | - |")
                continue
            ratio_label = f"**{ratio:.0%}**" if first_in_group else ""
            first_in_group = False
            print(f"| {ratio_label} | {method} | "
                  f"{m[1]['Recall']:.4f} | {m[3]['Recall']:.4f} | "
                  f"{m[5]['Recall']:.4f} | "
                  f"{m[5]['HR']:.4f} | {m[5]['NDCG']:.4f} | "
                  f"{m[5]['MRR']:.4f} |")

    kg_methods = ["TransE-Rec", "CB-KG", "KGAT", "KGIN"]
    print("\n\n📊 KG Methods Only (recommended for paper)")
    print("| Training | Method | Recall@1 | Recall@3 | Recall@5 | "
          "NDCG@5 | MRR@5 |")
    print("|---|---|---|---|---|---|---|")

    for ratio in SPARSITY_RATIOS:
        for method in kg_methods:
            m = all_results[ratio].get(method)
            if m is None:
                continue
            print(f"| {ratio:.0%} | {method} | "
                  f"{m[1]['Recall']:.4f} | {m[3]['Recall']:.4f} | "
                  f"{m[5]['Recall']:.4f} | "
                  f"{m[5]['NDCG']:.4f} | {m[5]['MRR']:.4f} |")

    json_out = {}
    for ratio in SPARSITY_RATIOS:
        rk = f"{ratio:.0%}"
        json_out[rk] = {}
        for method in METHOD_NAMES:
            m = all_results[ratio].get(method)
            if m:
                json_out[rk][method] = {
                    str(k): {met: float(v) for met, v in vals.items()}
                    for k, vals in m.items()
                }

    save_path = os.path.join(RESULT_DIR, 'sparsity_v3_results.json')
    with open(save_path, 'w') as f:
        json.dump(json_out, f, indent=2)


if __name__ == '__main__':
    main()