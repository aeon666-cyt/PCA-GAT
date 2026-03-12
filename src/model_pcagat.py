import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from config_pcagat import *


class ConstraintGate(nn.Module):
    def __init__(self, node_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or node_dim
        self.gate_net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.gate_net:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, src_emb, tgt_emb):
        gate_input = torch.cat([src_emb, tgt_emb], dim=-1)
        return self.gate_net(gate_input)


class PCAGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_relations, n_constraint_types=4,
                 dropout=0.3, negative_slope=0.2, use_constraint=True,
                 use_relation=True, use_gate=True,
                 use_simple_attention=False, use_residual=True,
                 use_bi_interaction=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_constraint = use_constraint
        self.use_relation = use_relation
        self.use_gate = use_gate and use_constraint
        self.use_residual = use_residual
        self.use_bi_interaction = use_bi_interaction

        self.W = nn.Linear(in_dim, out_dim, bias=False)

        if use_bi_interaction:
            self.W_bi = nn.Linear(out_dim, out_dim, bias=False)

        att_in_dim = out_dim * 2 + (in_dim if use_relation else 0)

        if use_simple_attention:
            self.attention = nn.Sequential(
                nn.Linear(att_in_dim, 1, bias=False),
                nn.LeakyReLU(negative_slope),
            )
        else:
            self.attention = nn.Sequential(
                nn.Linear(att_in_dim, out_dim),
                nn.LeakyReLU(negative_slope),
                nn.Linear(out_dim, 1, bias=False)
            )

        if use_constraint:
            self.lambda_c = nn.Parameter(
                torch.full((n_constraint_types,), CONSTRAINT_LAMBDA_INIT))
            self.lambda_global = nn.Parameter(
                torch.tensor(CONSTRAINT_LAMBDA_INIT))
            if self.use_gate:
                gate_hidden = CONSTRAINT_GATE_HIDDEN or out_dim
                self.constraint_gate = ConstraintGate(out_dim, gate_hidden)

        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        if self.use_bi_interaction and hasattr(self, 'W_bi'):
            nn.init.xavier_uniform_(self.W_bi.weight)
        for layer in self.attention:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)

    def get_bounded_lambdas(self):
        if USE_BOUNDED_LAMBDA:
            lc = torch.sigmoid(self.lambda_c) * (1.0 - LAMBDA_MIN) + LAMBDA_MIN
            lg = (torch.sigmoid(self.lambda_global)
                  * (LAMBDA_GLOBAL_SCALE - LAMBDA_GLOBAL_MIN)
                  + LAMBDA_GLOBAL_MIN)
        else:
            lc = self.lambda_c
            lg = self.lambda_global
        return lc, lg

    def forward(self, h, edge_index, edge_type, relation_emb,
                constraint_indices=None, constraint_values=None,
                constraint_types=None, return_attention=False):
        n_nodes = h.size(0)

        Wh = self.W(h)

        src_idx = edge_index[0]
        tgt_idx = edge_index[1]

        src_emb = Wh[src_idx]
        tgt_emb = Wh[tgt_idx]

        if self.use_relation:
            r_emb = relation_emb(edge_type)
            att_input = torch.cat([src_emb, tgt_emb, r_emb], dim=-1)
        else:
            att_input = torch.cat([src_emb, tgt_emb], dim=-1)

        e = self.attention(att_input).squeeze(-1)

        if self.use_constraint and constraint_indices is not None \
                and constraint_indices.size(1) > 0:
            e = self._apply_constraints(
                e, edge_index, n_nodes, Wh,
                constraint_indices, constraint_values, constraint_types
            )

        attention_weights = self._scatter_softmax(e, src_idx, n_nodes)
        attention_weights = self.attn_dropout(attention_weights)

        tgt_features = Wh[tgt_idx]
        weighted_features = attention_weights.unsqueeze(-1) * tgt_features

        side_emb = torch.zeros(n_nodes, self.out_dim, device=h.device)
        side_emb.scatter_add_(
            0, src_idx.unsqueeze(-1).expand_as(weighted_features),
            weighted_features)

        if self.use_bi_interaction:
            sum_out = self.leaky_relu(Wh + side_emb)
            bi_out = self.leaky_relu(self.W_bi(Wh * side_emb))
            h_new = sum_out + bi_out
        else:
            h_new = side_emb

        if self.use_residual:
            h_new = h_new + Wh

        h_new = F.elu(h_new)
        h_new = self.feat_dropout(h_new)

        if return_attention:
            return h_new, attention_weights
        return h_new

    def _apply_constraints(self, e, edge_index, n_nodes, Wh,
                           constraint_indices, constraint_values,
                           constraint_types):
        if constraint_indices is None or constraint_indices.size(1) == 0:
            return e

        c_src = constraint_indices[0]
        c_tgt = constraint_indices[1]
        edge_src = edge_index[0]
        edge_tgt = edge_index[1]

        lambda_c, lambda_global = self.get_bounded_lambdas()

        weighted_scores = torch.zeros(constraint_values.size(0), device=e.device)
        for ct in range(lambda_c.size(0)):
            mask = (constraint_types == ct).float()
            weighted_scores = weighted_scores + mask * lambda_c[ct] * constraint_values

        c_keys = c_src * n_nodes + c_tgt
        edge_keys = edge_src * n_nodes + edge_tgt

        max_key = n_nodes * n_nodes
        c_aggregated = torch.zeros(max_key, device=e.device,
                                   dtype=weighted_scores.dtype)
        c_aggregated = c_aggregated.scatter_add(0, c_keys, weighted_scores)
        c_exists = torch.zeros(max_key, device=e.device)
        c_exists.scatter_add_(0, c_keys, torch.ones_like(weighted_scores))

        matched_scores = c_aggregated[edge_keys]
        matched_mask = (c_exists[edge_keys] > 0)
        n_matched = matched_mask.sum().item()

        if n_matched == 0:
            avg_constraint_value = torch.mean(weighted_scores)
            global_bias = lambda_global * avg_constraint_value * 0.01
            e = e + global_bias
            return e

        matched_scores = matched_scores * matched_mask.float()

        if self.use_gate:
            if matched_mask.any():
                gate_src = Wh[edge_src[matched_mask]]
                gate_tgt = Wh[edge_tgt[matched_mask]]
                gate_vals = self.constraint_gate(gate_src, gate_tgt).squeeze(-1)
                gated_scores = matched_scores.clone()
                gated_scores[matched_mask] = gate_vals * matched_scores[matched_mask]
                matched_scores = gated_scores

        e = e + lambda_global * matched_scores
        return e

    def _scatter_softmax(self, scores, index, num_nodes):
        scores = scores.float()
        max_scores = torch.full((num_nodes,), -1e9, device=scores.device,
                                dtype=torch.float32)
        max_scores.scatter_reduce_(
            0, index, scores, reduce='amax', include_self=False)
        scores = scores - max_scores[index]
        exp_scores = torch.exp(scores.clamp(max=20))
        sum_exp = torch.zeros(num_nodes, device=scores.device, dtype=torch.float32)
        sum_exp.scatter_add_(0, index, exp_scores)
        return exp_scores / (sum_exp[index] + 1e-10)


class MultiHeadPCAGAT(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, n_relations,
                 n_constraint_types=4, dropout=0.3, negative_slope=0.2,
                 use_constraint=True, use_relation=True, use_gate=True,
                 use_simple_attention=False, use_residual=True,
                 use_bi_interaction=True,
                 concat=True):
        super().__init__()
        self.n_heads = n_heads
        self.concat = concat

        if concat:
            assert out_dim % n_heads == 0
            head_dim = out_dim // n_heads
        else:
            head_dim = out_dim

        self.heads = nn.ModuleList([
            PCAGATLayer(
                in_dim=in_dim, out_dim=head_dim,
                n_relations=n_relations,
                n_constraint_types=n_constraint_types,
                dropout=dropout, negative_slope=negative_slope,
                use_constraint=use_constraint,
                use_relation=use_relation,
                use_gate=use_gate,
                use_simple_attention=use_simple_attention,
                use_residual=use_residual,
                use_bi_interaction=use_bi_interaction,
            )
            for _ in range(n_heads)
        ])

    def forward(self, h, edge_index, edge_type, relation_emb,
                constraint_indices=None, constraint_values=None,
                constraint_types=None, return_attention=False):
        head_outputs = []
        all_attentions = []

        for head in self.heads:
            if return_attention:
                out, att = head(h, edge_index, edge_type, relation_emb,
                                constraint_indices, constraint_values,
                                constraint_types, return_attention=True)
                all_attentions.append(att)
            else:
                out = head(h, edge_index, edge_type, relation_emb,
                           constraint_indices, constraint_values,
                           constraint_types, return_attention=False)
            head_outputs.append(out)

        if self.concat:
            h_new = torch.cat(head_outputs, dim=-1)
        else:
            h_new = torch.mean(torch.stack(head_outputs), dim=0)

        if return_attention:
            avg_attention = torch.mean(torch.stack(all_attentions), dim=0)
            return h_new, avg_attention
        return h_new


class PCAGAT(nn.Module):
    def __init__(self, n_users, n_entities, n_relations, n_items, dim,
                 item2entity, edge_index, edge_type,
                 constraint_indices=None, constraint_values=None,
                 constraint_types=None,
                 n_layers=PCAGAT_LAYERS, n_heads=PCAGAT_HEADS,
                 dropout=PCAGAT_DROPOUT, use_constraint=True,
                 use_kg=True, use_multi_head=True,
                 use_gate=USE_CONSTRAINT_GATE,
                 use_simple_attention=USE_SIMPLE_ATTENTION,
                 use_residual=USE_RESIDUAL,
                 use_bi_interaction=USE_BI_INTERACTION,
                 layer_agg_mode=LAYER_AGG_MODE,
                 constraint_margin=CONSTRAINT_CONTRASTIVE_MARGIN,
                 device=DEVICE):
        super().__init__()

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.item2entity = item2entity
        self.device = device
        self.use_constraint = use_constraint
        self.use_kg = use_kg
        self.use_multi_head = use_multi_head
        self.use_gate = use_gate
        self.constraint_margin = constraint_margin
        self.layer_agg_mode = layer_agg_mode

        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_type', edge_type)

        if constraint_indices is not None:
            self.register_buffer('constraint_indices', constraint_indices)
            self.register_buffer('constraint_values', constraint_values)
            self.register_buffer('constraint_types', constraint_types)
        else:
            self.register_buffer('constraint_indices',
                                 torch.zeros(2, 0, dtype=torch.long))
            self.register_buffer('constraint_values',
                                 torch.zeros(0, dtype=torch.float))
            self.register_buffer('constraint_types',
                                 torch.zeros(0, dtype=torch.long))

        self.user_emb = nn.Embedding(n_users, dim)
        self.entity_emb = nn.Embedding(n_entities, dim)
        n_total_relations = n_relations * 2 + 1
        self.relation_emb = nn.Embedding(n_total_relations, dim)

        n_constraint_types = len(CONSTRAINT_TYPES)
        self.gat_layers = nn.ModuleList()

        for layer in range(n_layers):
            if use_multi_head and n_heads > 1:
                self.gat_layers.append(
                    MultiHeadPCAGAT(
                        in_dim=dim, out_dim=dim, n_heads=n_heads,
                        n_relations=n_total_relations,
                        n_constraint_types=n_constraint_types,
                        dropout=dropout,
                        use_constraint=use_constraint,
                        use_relation=use_kg,
                        use_gate=use_gate,
                        use_simple_attention=use_simple_attention,
                        use_residual=use_residual,
                        use_bi_interaction=use_bi_interaction,
                        concat=True
                    )
                )
            else:
                self.gat_layers.append(
                    PCAGATLayer(
                        in_dim=dim, out_dim=dim,
                        n_relations=n_total_relations,
                        n_constraint_types=n_constraint_types,
                        dropout=dropout,
                        use_constraint=use_constraint,
                        use_relation=use_kg,
                        use_gate=use_gate,
                        use_simple_attention=use_simple_attention,
                        use_residual=use_residual,
                        use_bi_interaction=use_bi_interaction,
                    )
                )

        if use_kg:
            self.kg_relation_emb = nn.Embedding(n_relations, dim)
            nn.init.xavier_uniform_(self.kg_relation_emb.weight)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(n_layers)
        ])

        if layer_agg_mode == 'concat':
            self.agg_proj = nn.Linear(dim * (n_layers + 1), dim, bias=False)
            nn.init.xavier_uniform_(self.agg_proj.weight)
        else:
            self.agg_proj = None

        self._init_weights()

        self._item_indices = torch.tensor(
            [item2entity.get(i, 0) for i in range(n_items)],
            dtype=torch.long, device=device
        )

        self._last_attention_weights = [None] * n_layers

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def load_pretrain(self, pretrain_user_emb, pretrain_item_emb, item2entity):
        with torch.no_grad():
            self.user_emb.weight.copy_(pretrain_user_emb.to(self.device))
            loaded = 0
            for item_id, item_emb in enumerate(pretrain_item_emb):
                entity_id = item2entity.get(item_id, None)
                if entity_id is not None and entity_id < self.n_entities:
                    self.entity_emb.weight[entity_id].copy_(
                        item_emb.to(self.device))
                    loaded += 1

    def _propagate(self, return_attention=False):
        all_emb = torch.cat(
            [self.user_emb.weight, self.entity_emb.weight], dim=0)
        emb_list = [all_emb]
        attention_list = []

        for layer_idx in range(self.n_layers):
            if return_attention:
                new_emb, att = self.gat_layers[layer_idx](
                    all_emb, self.edge_index, self.edge_type,
                    self.relation_emb,
                    self.constraint_indices, self.constraint_values,
                    self.constraint_types, return_attention=True
                )
                attention_list.append(att)
                self._last_attention_weights[layer_idx] = att.detach()
            else:
                new_emb = self.gat_layers[layer_idx](
                    all_emb, self.edge_index, self.edge_type,
                    self.relation_emb,
                    self.constraint_indices, self.constraint_values,
                    self.constraint_types, return_attention=False
                )

            new_emb = self.layer_norms[layer_idx](new_emb)

            if USE_CROSS_LAYER_RESIDUAL:
                all_emb = new_emb + all_emb
            else:
                all_emb = new_emb
            emb_list.append(all_emb)

        if self.layer_agg_mode == 'concat':
            final_emb = torch.cat(emb_list, dim=-1)
            final_emb = self.agg_proj(final_emb)
        else:
            final_emb = torch.mean(torch.stack(emb_list, dim=0), dim=0)

        if return_attention:
            return final_emb, attention_list
        return final_emb

    def forward(self, users, pos_items, neg_items):
        final_emb = self._propagate(return_attention=False)
        user_emb = final_emb[:self.n_users]
        entity_emb = final_emb[self.n_users:]
        item_emb = entity_emb[self._item_indices]
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        return pos_scores, neg_scores

    def predict(self, users):
        with torch.no_grad():
            final_emb = self._propagate(return_attention=False)
            user_emb = final_emb[:self.n_users]
            entity_emb = final_emb[self.n_users:]
            item_emb = entity_emb[self._item_indices]
            return torch.matmul(user_emb[users], item_emb.t())

    def predict_with_attention(self, users):
        with torch.no_grad():
            final_emb, attention_list = self._propagate(return_attention=True)
            user_emb = final_emb[:self.n_users]
            entity_emb = final_emb[self.n_users:]
            item_emb = entity_emb[self._item_indices]
            scores = torch.matmul(user_emb[users], item_emb.t())
            return scores, attention_list

    def compute_kg_loss(self, h, r, t, neg_t):
        if not self.use_kg:
            return torch.tensor(0.0, device=self.device)
        valid_mask = ((h < self.n_entities) & (t < self.n_entities) &
                      (neg_t < self.n_entities) & (r < self.n_relations))
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        h, r, t, neg_t = (h[valid_mask], r[valid_mask],
                          t[valid_mask], neg_t[valid_mask])
        h_emb = self.entity_emb(h)
        t_emb = self.entity_emb(t)
        neg_t_emb = self.entity_emb(neg_t)
        r_emb = self.kg_relation_emb(r)
        pos_score = torch.norm(h_emb + r_emb - t_emb, p=TRANSE_NORM, dim=1)
        neg_score = torch.norm(h_emb + r_emb - neg_t_emb, p=TRANSE_NORM, dim=1)
        return torch.mean(F.relu(pos_score - neg_score + KG_MARGIN))

    def compute_l2_loss(self, users, pos_items, neg_items):
        u_emb = self.user_emb(users)
        pos_emb = self.entity_emb(self._item_indices[pos_items])
        neg_emb = self.entity_emb(self._item_indices[neg_items])
        return (torch.mean(u_emb ** 2) + torch.mean(pos_emb ** 2) +
                torch.mean(neg_emb ** 2))

    def compute_constraint_alignment_loss(self, sample_size=2048):
        c_src = self.constraint_indices[0]
        c_tgt = self.constraint_indices[1]
        c_vals = self.constraint_values
        c_types = self.constraint_types
        if c_src.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        n_constraints = c_src.size(0)
        if n_constraints > sample_size:
            idx = torch.randperm(n_constraints, device=self.device)[:sample_size]
            c_src, c_tgt = c_src[idx], c_tgt[idx]
            c_vals, c_types = c_vals[idx], c_types[idx]
        all_emb = torch.cat([self.user_emb.weight, self.entity_emb.weight], dim=0)
        src_emb = all_emb[c_src].detach()
        tgt_emb = all_emb[c_tgt].detach()
        sim = F.cosine_similarity(src_emb, tgt_emb, dim=1)
        layer = self.gat_layers[0]
        if isinstance(layer, MultiHeadPCAGAT):
            lambda_c, lambda_global = layer.heads[0].get_bounded_lambdas()
        else:
            lambda_c, lambda_global = layer.get_bounded_lambdas()
        weighted_constraint = torch.zeros_like(c_vals)
        for ct in range(lambda_c.size(0)):
            mask = (c_types == ct).float()
            weighted_constraint = weighted_constraint + mask * lambda_c[ct]
        alignment = sim * c_vals * weighted_constraint
        return -lambda_global * torch.mean(alignment)

    def compute_constraint_contrastive_loss(self):
        c_src = self.constraint_indices[0]
        c_tgt = self.constraint_indices[1]
        c_vals = self.constraint_values
        if c_src.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        all_emb = torch.cat([self.user_emb.weight, self.entity_emb.weight], dim=0)
        src_emb = all_emb[c_src]
        tgt_emb = all_emb[c_tgt]
        dist = torch.norm(src_emb - tgt_emb, p=2, dim=1)
        pos_mask = c_vals > 0
        neg_mask = c_vals < 0
        loss = torch.tensor(0.0, device=self.device)
        if pos_mask.any():
            loss = loss + torch.mean(dist[pos_mask] * c_vals[pos_mask].abs())
        if neg_mask.any():
            push_loss = F.relu(self.constraint_margin - dist[neg_mask])
            loss = loss + torch.mean(push_loss * c_vals[neg_mask].abs())
        return loss

    def get_constraint_weights(self):
        weights = {}
        for layer_idx, layer in enumerate(self.gat_layers):
            if isinstance(layer, MultiHeadPCAGAT):
                for head_idx, head in enumerate(layer.heads):
                    key = f"layer{layer_idx}_head{head_idx}"
                    w = {}
                    if hasattr(head, 'lambda_c'):
                        lc, lg = head.get_bounded_lambdas()
                        w['lambda_c'] = lc.detach().cpu().numpy()
                        w['lambda_global'] = lg.detach().cpu().item()
                        w['lambda_c_raw'] = head.lambda_c.detach().cpu().numpy()
                        w['lambda_global_raw'] = head.lambda_global.detach().cpu().item()
                    if hasattr(head, 'constraint_gate'):
                        w['has_gate'] = True
                        w['gate_param_norms'] = {
                            n: p.detach().cpu().norm().item()
                            for n, p in head.constraint_gate.named_parameters()
                        }
                    if w:
                        weights[key] = w
            elif isinstance(layer, PCAGATLayer):
                key = f"layer{layer_idx}"
                w = {}
                if hasattr(layer, 'lambda_c'):
                    lc, lg = layer.get_bounded_lambdas()
                    w['lambda_c'] = lc.detach().cpu().numpy()
                    w['lambda_global'] = lg.detach().cpu().item()
                    w['lambda_c_raw'] = layer.lambda_c.detach().cpu().numpy()
                    w['lambda_global_raw'] = layer.lambda_global.detach().cpu().item()
                if hasattr(layer, 'constraint_gate'):
                    w['has_gate'] = True
                if w:
                    weights[key] = w
        return weights


class BPRMF_Pretrain(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, u, i, neg_i):
        u_e = self.user_emb(u)
        i_e = self.item_emb(i)
        neg_i_e = self.item_emb(neg_i)
        return torch.sum(u_e * i_e, dim=1), torch.sum(u_e * neg_i_e, dim=1)

    def predict(self, u):
        return torch.matmul(self.user_emb(u), self.item_emb.weight.t())


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
        graph = self.graph.float()
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        final_emb = torch.mean(torch.stack(embs, dim=1), dim=1)
        return torch.split(final_emb, [self.n_users, self.n_items])

    def forward(self, u, i, neg_i):
        u_all, i_all = self.computer()
        return (torch.sum(u_all[u] * i_all[i], dim=1),
                torch.sum(u_all[u] * i_all[neg_i], dim=1))

    def predict(self, u):
        u_all, i_all = self.computer()
        return torch.matmul(u_all[u], i_all.t())


class StandardGAT(nn.Module):
    def __init__(self, n_users, n_entities, n_relations, n_items, dim,
                 item2entity, edge_index, edge_type,
                 n_layers=2, n_heads=4, dropout=0.3, device=DEVICE):
        super().__init__()
        self.model = PCAGAT(
            n_users=n_users, n_entities=n_entities,
            n_relations=n_relations, n_items=n_items, dim=dim,
            item2entity=item2entity, edge_index=edge_index,
            edge_type=edge_type,
            constraint_indices=None, constraint_values=None,
            constraint_types=None,
            n_layers=n_layers, n_heads=n_heads, dropout=dropout,
            use_constraint=False, use_kg=True, use_gate=False,
            use_multi_head=True, device=device
        )

    def forward(self, u, i, neg_i):
        return self.model(u, i, neg_i)

    def predict(self, u):
        return self.model.predict(u)

    def compute_kg_loss(self, h, r, t, neg_t):
        return self.model.compute_kg_loss(h, r, t, neg_t)

    def compute_l2_loss(self, u, i, neg_i):
        return self.model.compute_l2_loss(u, i, neg_i)