import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import random
from collections import defaultdict

from config_pcagat import BATCH_SIZE, KG_BATCH_SIZE


def _collate_fn(batch):
    return torch.stack(batch, dim=0)


class KGDataset(Dataset):
    def __init__(self, triplets, n_ent):
        triplets = np.array(triplets)
        h_valid = (triplets[:, 0] >= 0) & (triplets[:, 0] < n_ent)
        t_valid = (triplets[:, 2] >= 0) & (triplets[:, 2] < n_ent)
        valid_mask = h_valid & t_valid

        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()

        self.triplets = torch.LongTensor(triplets[valid_mask])
        self.n_ent = n_ent

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        h, r, t = self.triplets[idx]
        neg_t = torch.randint(0, self.n_ent, (1,)).item()
        while neg_t == t:
            neg_t = torch.randint(0, self.n_ent, (1,)).item()
        return torch.LongTensor([h, r, t, neg_t])


class InterDataset(Dataset):
    def __init__(self, data, n_items, train_set, test_dict=None):
        self.data = torch.LongTensor(data)
        self.n_items = n_items
        self.train_set = train_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i = self.data[idx]
        u_val = u.item() if isinstance(u, torch.Tensor) else u
        neg_i = torch.randint(0, self.n_items, (1,)).item()
        while neg_i in self.train_set.get(u_val, set()):
            neg_i = torch.randint(0, self.n_items, (1,)).item()
        return torch.LongTensor([u, i, neg_i])


class ConstraintAwareInterDataset(Dataset):
    def __init__(self, data, n_items, train_set, user_hard_negs,
                 hard_neg_ratio=0.3, test_dict=None):
        self.data = torch.LongTensor(data)
        self.n_items = n_items
        self.train_set = train_set
        self.user_hard_negs = user_hard_negs
        self.hard_neg_ratio = hard_neg_ratio
        self.test_dict = test_dict if test_dict is not None else {}

        self.user_neg_pools = {}
        all_items = set(range(n_items))
        for u in train_set.keys():
            train_items = set(train_set[u])
            test_items = set(self.test_dict.get(u, []))
            exclude = train_items | test_items
            neg_pool = list(all_items - exclude)
            if len(neg_pool) == 0:
                neg_pool = list(all_items - train_items)
            self.user_neg_pools[u] = neg_pool

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i = self.data[idx]
        u_val = u.item() if isinstance(u, torch.Tensor) else u

        if (u_val in self.user_hard_negs
                and len(self.user_hard_negs[u_val]) > 0
                and random.random() < self.hard_neg_ratio):
            valid_hard_negs = [
                item for item in self.user_hard_negs[u_val]
                if item not in self.test_dict.get(u_val, set())
            ]
            if valid_hard_negs:
                neg_i = random.choice(valid_hard_negs)
            else:
                neg_i = self._random_neg(u_val)
            if neg_i in self.train_set.get(u_val, set()):
                neg_i = self._random_neg(u_val)
        else:
            neg_i = self._random_neg(u_val)

        return torch.LongTensor([u, i, neg_i])

    def _random_neg(self, u_val):
        neg_pool = self.user_neg_pools.get(u_val)
        if neg_pool and len(neg_pool) > 0:
            return np.random.choice(neg_pool)
        else:
            neg_i = torch.randint(0, self.n_items, (1,)).item()
            while neg_i in self.train_set.get(u_val, set()):
                neg_i = torch.randint(0, self.n_items, (1,)).item()
            return neg_i


class MachiningDataset:
    def __init__(self, path, device='cuda'):
        self.path = path
        self.device = device

        user_list, self.user_names = self._load_id_and_names(
            os.path.join(path, 'user_list.txt'))
        item_list, self.item_names = self._load_id_and_names(
            os.path.join(path, 'item_list.txt'))
        entity_list, self.entity_name_list = self._load_id_and_names(
            os.path.join(path, 'entity_list.txt'))
        relation_list, self.relation_name_list = self._load_id_and_names(
            os.path.join(path, 'relation_list.txt'))

        self.train_data = self._load_variable_inter_file(
            os.path.join(path, 'train.txt'))
        self.test_data = self._load_variable_inter_file(
            os.path.join(path, 'test.txt'))
        self.kg_data = np.loadtxt(
            os.path.join(path, 'kg_final.txt'), dtype=np.int32)

        self.n_users = (len(user_list) if user_list
                        else int(self.train_data[:, 0].max() + 1))
        self.n_items = (len(item_list) if item_list
                        else int(max(self.train_data[:, 1].max(),
                                     self.test_data[:, 1].max()) + 1))
        self.n_relations = (len(relation_list) if relation_list
                            else int(self.kg_data[:, 1].max() + 1))

        kg_entities = set(self.kg_data[:, 0]) | set(self.kg_data[:, 2])
        self.n_entities = (max(max(kg_entities) + 1, len(entity_list))
                           if kg_entities else len(entity_list))

        self.test_dict = defaultdict(list)
        for u, i in self.test_data:
            self.test_dict[u].append(i)

        self.train_set = defaultdict(set)
        for u, i in self.train_data:
            self.train_set[u].add(i)

        self.task_adopted = self.train_set

        DEBUG_MODE = False

        if DEBUG_MODE:
            valid_users = set(range(5000))
            self.train_data = self.train_data[
                np.isin(self.train_data[:, 0], list(valid_users))
            ]
            self.test_data = self.test_data[
                np.isin(self.test_data[:, 0], list(valid_users))
            ]
            self.n_users = 5000

            self.test_dict = defaultdict(list)
            for u, i in self.test_data:
                self.test_dict[u].append(i)
            self.train_set = defaultdict(set)
            for u, i in self.train_data:
                self.train_set[u].add(i)
            self.task_adopted = self.train_set

            self.kg_data = self.kg_data[:50000]

        self._build_kg_adjacency()
        self._build_item_entity_mapping()

        self.constraint_rules = self._load_constraint_rules()

        self.entity_names = self._load_json(
            os.path.join(path, 'entity_names.json'))
        self.relation_names = self._load_json(
            os.path.join(path, 'relation_names.json'))

        self.user_hard_negs = {}
        self._build_hard_neg_pools()

    def subsample_train(self, ratio=0.6):
        np.random.seed(42)
        n_original = len(self.train_data)
        n_keep = int(n_original * ratio)
        idx = np.random.choice(n_original, n_keep, replace=False)
        self.train_data = self.train_data[idx]
        self.train_set = defaultdict(set)
        for u, i in self.train_data:
            self.train_set[u].add(i)
        self.task_adopted = self.train_set

    def inject_noise(self, noise_ratio=0.1):
        np.random.seed(42)
        n_noise = int(len(self.train_data) * noise_ratio)
        noise = np.column_stack([
            np.random.randint(0, self.n_users, n_noise),
            np.random.randint(0, self.n_items, n_noise)
        ]).astype(np.int32)
        self.train_data = np.vstack([self.train_data, noise])

    def _load_id_and_names(self, filepath):
        ids = []
        names = {}
        if not os.path.exists(filepath):
            return [], {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if i == 0 and not parts[-1].isdigit():
                    continue
                try:
                    idx = int(parts[-1])
                    name = parts[0] if len(parts) >= 2 else f"id_{idx}"
                    ids.append(idx)
                    names[idx] = name
                except (ValueError, IndexError):
                    continue
        return ids, names

    def _load_variable_inter_file(self, filepath):
        data_list = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    u_id = int(parts[0])
                    for i_id in parts[1:]:
                        data_list.append([u_id, int(i_id)])
        return np.array(data_list, dtype=np.int32)

    def _load_constraint_rules(self):
        filepath = os.path.join(self.path, 'constraint_rules.json')
        if not os.path.exists(filepath):
            alt_path = os.path.join(self.path, 'constraint_matrix.txt')
            if os.path.exists(alt_path):
                return self._load_constraint_matrix_txt(alt_path)
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_constraint_matrix_txt(self, filepath):
        type_names = ['material_operation', 'precision_operation',
                      'operation_sequence', 'feature_operation']
        rules = {'entity_type_constraints': {t: {} for t in type_names},
                 'default_score': 0.0}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    ent_i, ent_j = int(parts[0]), int(parts[1])
                    c_type, score = int(parts[2]), float(parts[3])
                    if c_type < len(type_names):
                        key = f"{ent_i},{ent_j}"
                        rules['entity_type_constraints'][type_names[c_type]][key] = score
        return rules

    def _load_json(self, filepath):
        if not os.path.exists(filepath):
            return {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}
        except Exception:
            return {}

    def _build_kg_adjacency(self):
        self.kg_dict = defaultdict(list)
        for h, r, t in self.kg_data:
            self.kg_dict[h].append((r, t))
            self.kg_dict[t].append((r + self.n_relations, h))
        self.n_relations_with_inverse = self.n_relations * 2

    def _build_item_entity_mapping(self):
        mapping_file = os.path.join(self.path, 'item_index2entity_id.txt')
        if os.path.exists(mapping_file):
            self.item2entity = {}
            with open(mapping_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        item_id, entity_id = map(int, parts)
                        self.item2entity[item_id] = entity_id
        else:
            self.item2entity = {i: i for i in range(self.n_items)}

        for i in range(self.n_items):
            if i not in self.item2entity:
                self.item2entity[i] = i

    def _build_hard_neg_pools(self):
        entity_constraints = self.constraint_rules.get(
            'entity_type_constraints', {})
        if not entity_constraints:
            return

        neg_pairs = set()
        for ctype, pairs in entity_constraints.items():
            for pair_key, score in pairs.items():
                if score < 0:
                    pk = pair_key.split(',')
                    if len(pk) == 2:
                        neg_pairs.add((int(pk[0]), int(pk[1])))

        if not neg_pairs:
            return

        item_neighbor_ents = {}
        for item_id in range(self.n_items):
            ent_id = self.item2entity.get(item_id)
            if ent_id is not None:
                neighbors = {ent_id}
                for r, t in self.kg_dict.get(ent_id, []):
                    if t < self.n_entities:
                        neighbors.add(t)
                item_neighbor_ents[item_id] = neighbors

        for u in range(self.n_users):
            user_entities = set()
            for item in self.train_set.get(u, []):
                ent_id = self.item2entity.get(item)
                if ent_id is not None:
                    user_entities.add(ent_id)
                    for r, t in self.kg_dict.get(ent_id, []):
                        if t < self.n_entities:
                            user_entities.add(t)

            if not user_entities:
                continue

            hard_negs = []
            for item_id in range(self.n_items):
                if item_id in self.train_set.get(u, set()):
                    continue
                item_ents = item_neighbor_ents.get(item_id, set())
                violation = False
                for ue in user_entities:
                    for ie in item_ents:
                        if (ue, ie) in neg_pairs or (ie, ue) in neg_pairs:
                            violation = True
                            break
                    if violation:
                        break
                if violation:
                    hard_negs.append(item_id)

            if hard_negs:
                self.user_hard_negs[u] = hard_negs

    def get_kg_loader(self, batch_size=KG_BATCH_SIZE):
        return DataLoader(
            KGDataset(self.kg_data, self.n_entities),
            batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )

    def get_interaction_loader(self, batch_size=BATCH_SIZE, use_hard_neg=False,
                               hard_neg_ratio=0.7):
        if use_hard_neg and self.user_hard_negs:
            dataset = ConstraintAwareInterDataset(
                self.train_data, self.n_items, self.train_set,
                self.user_hard_negs, hard_neg_ratio, self.test_dict
            )
        else:
            dataset = InterDataset(self.train_data, self.n_items,
                                   self.train_set, self.test_dict)

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True,
            collate_fn=_collate_fn
        )

    def get_sparse_graph(self):
        import scipy.sparse as sp
        n_nodes = self.n_users + self.n_items
        users_np = np.array([u for u, i in self.train_data])
        items_np = np.array([i for u, i in self.train_data])
        row_idx = np.concatenate([users_np, items_np + self.n_users])
        col_idx = np.concatenate([items_np + self.n_users, users_np])
        data = np.ones(len(row_idx), dtype=np.float32)
        adj_mat = sp.coo_matrix((data, (row_idx, col_idx)),
                                shape=(n_nodes, n_nodes))
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum + 1e-8, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = sp.diags(d_inv_sqrt)
        norm_adj = (d_mat @ adj_mat @ d_mat).tocoo()
        indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        graph = torch.sparse_coo_tensor(indices, values,
                                        torch.Size(norm_adj.shape))
        return graph.to(self.device)

    def get_edge_data(self):
        edges_h, edges_t, edges_r = [], [], []
        interact_relation = self.n_relations * 2

        for u, items in self.train_set.items():
            for i in items:
                entity_id = self.item2entity.get(i, None)
                if entity_id is not None:
                    edges_h.append(u)
                    edges_t.append(self.n_users + entity_id)
                    edges_r.append(interact_relation)
                    edges_h.append(self.n_users + entity_id)
                    edges_t.append(u)
                    edges_r.append(interact_relation)

        n_interact = len(edges_h)

        kg_sample = self.kg_data

        for h, r, t in kg_sample:
            edges_h.append(self.n_users + h)
            edges_t.append(self.n_users + t)
            edges_r.append(r)
            edges_h.append(self.n_users + t)
            edges_t.append(self.n_users + h)
            edges_r.append(r + self.n_relations)

        n_kg = len(edges_h) - n_interact
        edge_index = torch.LongTensor([edges_h, edges_t])
        edge_type = torch.LongTensor(edges_r)

        return edge_index, edge_type

    def _expand_constraint_lookup(self, decay=0.5):
        if hasattr(self, '_cached_expanded_lookup'):
            return self._cached_expanded_lookup, self._cached_expanded_types

        entity_constraints = self.constraint_rules.get(
            'entity_type_constraints', {})
        type_map = {
            'material_operation': 0,
            'precision_operation': 1,
            'operation_sequence': 2,
            'feature_operation': 3,
        }
        directional_types = {'operation_sequence'}
        lookup = {}
        type_info = {}

        for ctype, pairs in entity_constraints.items():
            ctype_id = type_map.get(ctype, 0)
            is_directional = ctype in directional_types
            for pair_key, score in pairs.items():
                pk = pair_key.split(',')
                if len(pk) != 2:
                    continue
                a, b = int(pk[0]), int(pk[1])
                lookup[(a, b)] = score
                type_info[(a, b)] = ctype_id
                if not is_directional and (b, a) not in lookup:
                    lookup[(b, a)] = score
                    type_info[(b, a)] = ctype_id

        n_after_bidir = len(lookup)
        propagated = {}

        for (a, b), score in list(lookup.items()):
            ctype_id = type_info[(a, b)]
            for r, neighbor in self.kg_dict.get(a, []):
                if neighbor < self.n_entities and (neighbor, b) not in lookup:
                    prop_score = score * decay
                    if abs(prop_score) > 0.1:
                        key = (neighbor, b)
                        if key not in propagated or abs(prop_score) > abs(propagated[key]):
                            propagated[key] = prop_score
                            type_info[key] = ctype_id
            for r, neighbor in self.kg_dict.get(b, []):
                if neighbor < self.n_entities and (a, neighbor) not in lookup:
                    prop_score = score * decay
                    if abs(prop_score) > 0.1:
                        key = (a, neighbor)
                        if key not in propagated or abs(prop_score) > abs(propagated[key]):
                            propagated[key] = prop_score
                            type_info[key] = ctype_id

        for key, score in propagated.items():
            if key not in lookup:
                lookup[key] = score

        n_after_prop = len(lookup)
        n_orig = sum(len(entity_constraints.get(t, {}))
                     for t in ['material_operation', 'precision_operation',
                                'operation_sequence', 'feature_operation'])

        self._cached_expanded_lookup = lookup
        self._cached_expanded_types = type_info
        return lookup, type_info

    def build_full_constraint_matrix(self, n_nodes):
        expanded_lookup, expanded_types = self._expand_constraint_lookup()
        src_ids, tgt_ids, values, types = [], [], [], []

        for (ent_a, ent_b), score in expanded_lookup.items():
            node_a = self.n_users + ent_a
            node_b = self.n_users + ent_b
            if node_a < n_nodes and node_b < n_nodes:
                src_ids.append(node_a)
                tgt_ids.append(node_b)
                values.append(score)
                types.append(expanded_types.get((ent_a, ent_b), 0))

        if src_ids:
            indices = torch.LongTensor([src_ids, tgt_ids])
            vals = torch.FloatTensor(values)
            typs = torch.LongTensor(types)
        else:
            indices = torch.zeros(2, 0, dtype=torch.long)
            vals = torch.zeros(0)
            typs = torch.zeros(0, dtype=torch.long)

        n_types = len(set(types)) if types else 0
        return indices, vals, typs

    def build_constraint_scores(self, edge_index, edge_type):
        n_edges = edge_index.size(1)
        scores = torch.zeros(n_edges)
        if not self.constraint_rules:
            return scores

        expanded_lookup, _ = self._expand_constraint_lookup()
        default_score = self.constraint_rules.get('default_score', 0.0)
        h_nodes = edge_index[0].numpy()
        t_nodes = edge_index[1].numpy()
        r_types = edge_type.numpy()
        interact_relation = self.n_relations * 2
        n_assigned = 0

        for idx in range(n_edges):
            h, t, r = h_nodes[idx], t_nodes[idx], r_types[idx]
            if r == interact_relation:
                continue
            h_entity = h - self.n_users if h >= self.n_users else -1
            t_entity = t - self.n_users if t >= self.n_users else -1
            if h_entity >= 0 and t_entity >= 0:
                score = expanded_lookup.get((h_entity, t_entity), default_score)
                if score != 0:
                    scores[idx] = score
                    n_assigned += 1

        return scores

    def get_part_name(self, part_id):
        return self.user_names.get(part_id, f"Part_{part_id}")

    def get_plan_name(self, plan_id):
        return self.item_names.get(plan_id, f"Plan_{plan_id}")

    def get_entity_name(self, entity_id):
        if entity_id in self.entity_names:
            return self.entity_names[entity_id]
        if entity_id in self.entity_name_list:
            return self.entity_name_list[entity_id]
        return f"Entity_{entity_id}"

    def get_relation_name(self, relation_id):
        if relation_id in self.relation_names:
            return self.relation_names[relation_id]
        if relation_id in self.relation_name_list:
            return self.relation_name_list[relation_id]
        if (relation_id >= self.n_relations
                and relation_id < self.n_relations * 2):
            base_name = self.get_relation_name(relation_id - self.n_relations)
            return f"inv_{base_name}"
        if relation_id == self.n_relations * 2:
            return "interact"
        return f"Relation_{relation_id}"

    def get_entity_names(self):
        return self.entity_names

    def get_relation_names(self):
        return self.relation_names


def load_dataset():
    from config_pcagat import DATASET_PATH, DEVICE, USE_AMAZON
    device_str = str(DEVICE)
    return MachiningDataset(DATASET_PATH, device=device_str)