import torch
import numpy as np
from collections import defaultdict
import heapq

try:
    from config_pcagat import *
except ImportError:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EXPLAIN_TOP_K_PATHS = 3
    EXPLAIN_MAX_HOP = 3
    CONSTRAINT_TYPES = [
        'material_operation',
        'precision_operation',
        'operation_sequence',
        'feature_operation'
    ]


class PCAGATExplainer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.n_users = dataset.n_users
        self._build_adjacency()
        self._build_constraint_edge_set()

    def _build_adjacency(self):
        edge_index = self.model.edge_index.cpu()
        edge_type = self.model.edge_type.cpu()

        self.adj = defaultdict(list)
        seen_edges = set()

        for idx in range(edge_index.size(1)):
            src = edge_index[0, idx].item()
            tgt = edge_index[1, idx].item()
            rel = edge_type[idx].item()

            if (src, tgt, rel) not in seen_edges:
                self.adj[src].append((tgt, rel, idx))
                seen_edges.add((src, tgt, rel))

            if (tgt, src, rel) not in seen_edges:
                self.adj[tgt].append((src, rel, idx))
                seen_edges.add((tgt, src, rel))

        print(f"   Built adjacency: {len(self.adj)} nodes with neighbors")

    def _build_constraint_edge_set(self):
        self.constraint_edge_set = set()
        self.constraint_edge_types = {}
        self.constraint_influenced_set = set()
        self.constraint_influenced_types = {}

        c_indices = self.model.constraint_indices.cpu()
        c_types = self.model.constraint_types.cpu()
        c_values = self.model.constraint_values.cpu()

        if c_indices.size(1) == 0:
            return

        edge_index = self.model.edge_index.cpu()

        c_lookup = {}
        for k in range(c_indices.size(1)):
            key = (c_indices[0, k].item(), c_indices[1, k].item())
            c_lookup[key] = (c_types[k].item(), c_values[k].item())

        self.constrained_nodes = set()
        self.node_constraint_types = defaultdict(set)

        for k in range(c_indices.size(1)):
            src_node = c_indices[0, k].item()
            tgt_node = c_indices[1, k].item()
            ctype = c_types[k].item()
            self.constrained_nodes.add(src_node)
            self.constrained_nodes.add(tgt_node)
            self.node_constraint_types[src_node].add(ctype)
            self.node_constraint_types[tgt_node].add(ctype)

        interact_relation = self.dataset.n_relations * 2
        edge_type = self.model.edge_type.cpu()

        for idx in range(edge_index.size(1)):
            src = edge_index[0, idx].item()
            tgt = edge_index[1, idx].item()
            rel = edge_type[idx].item()

            if (src, tgt) in c_lookup:
                self.constraint_edge_set.add(idx)
                ctype, cval = c_lookup[(src, tgt)]
                self.constraint_edge_types[idx] = {
                    'type_id': ctype,
                    'type_name': CONSTRAINT_TYPES[ctype]
                    if ctype < len(CONSTRAINT_TYPES)
                    else f'type_{ctype}',
                    'value': cval,
                    'match_level': 'direct'
                }

            elif rel != interact_relation:
                src_constrained = src in self.constrained_nodes
                tgt_constrained = tgt in self.constrained_nodes
                if src_constrained or tgt_constrained:
                    self.constraint_influenced_set.add(idx)
                    src_types = self.node_constraint_types.get(src, set())
                    tgt_types = self.node_constraint_types.get(tgt, set())
                    common_types = src_types & tgt_types
                    involved_types = common_types if common_types else (src_types | tgt_types)
                    primary_type = min(involved_types) if involved_types else 0
                    self.constraint_influenced_types[idx] = {
                        'type_id': primary_type,
                        'type_name': CONSTRAINT_TYPES[primary_type]
                        if primary_type < len(CONSTRAINT_TYPES)
                        else f'type_{primary_type}',
                        'involved_types': list(involved_types),
                        'match_level': 'influenced',
                        'src_constrained': src_constrained,
                        'tgt_constrained': tgt_constrained,
                    }

        n_direct = len(self.constraint_edge_set)
        n_influenced = len(self.constraint_influenced_set)
        n_total_edges = edge_index.size(1)
        print(
            f"   Constraint edge matching: "
            f"direct={n_direct}, influenced={n_influenced}, "
            f"total_edges={n_total_edges}"
        )

    def _get_attention_modulation(self):
        if hasattr(self, '_cached_modulation'):
            return self._cached_modulation

        model = self.model
        model.eval()

        if (
            not hasattr(model, 'constraint_values')
            or model.constraint_values.size(0) == 0
            or not model.use_constraint
        ):
            print("   Model has no constraint mechanism")
            self._cached_modulation = None
            return None

        with torch.no_grad():
            _, attn_with_list = model._propagate(return_attention=True)

            orig_values = model.constraint_values.clone()
            model.constraint_values.zero_()
            _, attn_without_list = model._propagate(return_attention=True)

            model.constraint_values.copy_(orig_values)

        attn_with = torch.zeros_like(attn_with_list[0])
        attn_without = torch.zeros_like(attn_without_list[0])

        for aw, awo in zip(attn_with_list, attn_without_list):
            attn_with += aw
            attn_without += awo

        attn_with /= len(attn_with_list)
        attn_without /= len(attn_without_list)

        attn_with_np = attn_with.cpu().numpy()
        attn_without_np = attn_without.cpu().numpy()

        diff = attn_with_np - attn_without_np
        abs_diff = np.abs(diff)
        base = np.abs(attn_without_np) + 1e-10
        rel_modulation = abs_diff / base

        edge_type = model.edge_type.cpu().numpy()
        interact_rel = self.dataset.n_relations * 2
        kg_mask = edge_type != interact_rel

        direct_mask = np.zeros(len(diff), dtype=bool)
        for eidx in self.constraint_edge_set:
            if eidx < len(diff):
                direct_mask[eidx] = True

        global_cam = float(np.mean(rel_modulation))
        kg_edge_cam = float(np.mean(rel_modulation[kg_mask])) if kg_mask.any() else 0.0
        constrained_cam = float(np.mean(rel_modulation[direct_mask])) if direct_mask.any() else 0.0

        self._cached_modulation = {
            'attn_with': attn_with_np,
            'attn_without': attn_without_np,
            'diff': diff,
            'abs_diff': abs_diff,
            'rel_modulation': rel_modulation,
            'global_cam': global_cam,
            'kg_edge_cam': kg_edge_cam,
            'constrained_cam': constrained_cam,
        }

        n_modulated = int((abs_diff > 1e-6).sum())
        print(
            f"   CAM computed: {n_modulated}/{len(diff)} edges modulated; "
            f"global_CAM={global_cam:.4f}, "
            f"KG_edge_CAM={kg_edge_cam:.4f}, "
            f"constrained_CAM={constrained_cam:.4f}"
        )

        return self._cached_modulation

    def _get_score_modulation(self):
        if hasattr(self, '_cached_score_mod'):
            return self._cached_score_mod

        model = self.model
        model.eval()

        if (
            not hasattr(model, 'constraint_values')
            or model.constraint_values.size(0) == 0
            or not model.use_constraint
        ):
            self._cached_score_mod = None
            return None

        with torch.no_grad():
            all_parts = torch.arange(self.n_users).to(DEVICE)
            scores_with, _ = model.predict_with_attention(all_parts)
            scores_with = scores_with.cpu()

            orig_values = model.constraint_values.clone()
            model.constraint_values.zero_()
            scores_without, _ = model.predict_with_attention(all_parts)
            scores_without = scores_without.cpu()

            model.constraint_values.copy_(orig_values)

        self._cached_score_mod = {
            'scores_with': scores_with,
            'scores_without': scores_without,
        }

        diff = (scores_with - scores_without).abs()
        base = scores_without.abs() + 1e-10
        rel_change = (diff / base).mean().item()
        print(f"   RSM computed: mean relative score change = {rel_change:.4f}")

        return self._cached_score_mod

    def get_pair_rsm(self, part_id, plan_id):
        mod = self._get_score_modulation()
        if mod is None:
            return 0.0, 0.0, 0.0

        s_with = mod['scores_with'][part_id, plan_id].item()
        s_without = mod['scores_without'][part_id, plan_id].item()
        rsm = abs(s_with - s_without) / (abs(s_without) + 1e-10)
        return rsm, s_with, s_without

    def compute_rsm_metrics(self, test_pairs):
        mod = self._get_score_modulation()
        if mod is None:
            return {
                'rsm_mean': 0.0,
                'rsm_std': 0.0,
                'rank_change_mean': 0.0,
                'rank_degraded': 0
            }

        rsm_values = []
        rank_changes = []
        n_degraded = 0

        for part_id, plan_id in test_pairs:
            rsm, s_with, s_without = self.get_pair_rsm(part_id, plan_id)
            rsm_values.append(rsm)

            rank_with = int((mod['scores_with'][part_id] >= s_with).sum().item())
            rank_without = int((mod['scores_without'][part_id] >= s_without).sum().item())
            rank_change = rank_without - rank_with
            rank_changes.append(rank_change)

            if rank_change > 0:
                n_degraded += 1

        return {
            'rsm_mean': float(np.mean(rsm_values)),
            'rsm_std': float(np.std(rsm_values)),
            'rsm_max': float(np.max(rsm_values)),
            'rsm_min': float(np.min(rsm_values)),
            'rank_change_mean': float(np.mean(rank_changes)),
            'rank_degraded': n_degraded,
            'rank_degraded_ratio': n_degraded / len(test_pairs) if test_pairs else 0,
            'n_pairs': len(test_pairs),
        }

    def explain_recommendation(
        self,
        part_id,
        plan_id,
        top_k=EXPLAIN_TOP_K_PATHS,
        max_hop=EXPLAIN_MAX_HOP
    ):
        self.model.eval()

        with torch.no_grad():
            scores, attention_list = self.model.predict_with_attention(
                torch.tensor([part_id]).to(DEVICE)
            )

        rec_score = scores[0, plan_id].item()

        part_node = part_id
        plan_entity = self.dataset.item2entity.get(plan_id, plan_id)
        plan_node = self.n_users + plan_entity

        paths = self._find_attention_paths(
            part_node,
            plan_node,
            attention_list,
            max_hop,
            max_paths=top_k * 10
        )

        paths = self._diverse_rerank(paths, top_k)
        formatted_paths = [self._format_path(p) for p in paths]
        constraint_info = self._analyze_path_constraints(paths)
        gate_info = self._get_gate_values_for_recommendation(part_id, plan_id, paths)
        rsm, score_with, score_without = self.get_pair_rsm(part_id, plan_id)

        return {
            'part': self.dataset.get_part_name(part_id),
            'part_id': int(part_id),
            'plan': self.dataset.get_plan_name(plan_id),
            'plan_id': int(plan_id),
            'score': float(rec_score),
            'rsm': float(rsm),
            'score_without_constraint': float(score_without),
            'paths': formatted_paths,
            'constraint_info': constraint_info,
            'gate_info': gate_info,
        }

    def _find_attention_paths(self, source, target, attention_list, max_hop, max_paths=20):
        if not attention_list or attention_list[0] is None:
            return []

        avg_attention = torch.zeros_like(attention_list[0])
        for att in attention_list:
            avg_attention += att
        avg_attention /= len(attention_list)
        avg_attention = avg_attention.cpu().numpy()

        found_paths = []
        visited_paths = set()

        queue = [(-1.0, source, [source], [], [], 0)]
        max_iterations = 50000
        iteration = 0

        while queue and len(found_paths) < max_paths and iteration < max_iterations:
            iteration += 1
            neg_score, current, path_nodes, path_edges, path_rels, hop = heapq.heappop(queue)
            score = -neg_score

            if current == target and hop > 0:
                path_key = tuple(path_nodes)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    found_paths.append({
                        'nodes': list(path_nodes),
                        'edges': list(path_edges),
                        'relations': list(path_rels),
                        'score': score,
                        'hops': hop
                    })
                continue

            if hop >= max_hop:
                continue

            for neighbor, relation, edge_idx in self.adj.get(current, []):
                if neighbor in path_nodes:
                    continue

                if edge_idx < len(avg_attention):
                    edge_att = float(avg_attention[edge_idx])
                else:
                    edge_att = 0.001

                edge_att = max(edge_att, 1e-10)
                new_score = score * edge_att

                if new_score < 1e-50:
                    continue

                heapq.heappush(
                    queue,
                    (
                        -new_score,
                        neighbor,
                        path_nodes + [neighbor],
                        path_edges + [edge_idx],
                        path_rels + [relation],
                        hop + 1
                    )
                )

        if not found_paths:
            found_paths = self._bfs_fallback(source, target, avg_attention, max_hop, max_paths)

        return found_paths

    def _bfs_fallback(self, source, target, avg_attention, max_hop, max_paths=5):
        from collections import deque

        queue = deque([(source, [source], [], [], 0)])
        visited = {source}
        found_paths = []

        while queue and len(found_paths) < max_paths:
            current, path_nodes, path_edges, path_rels, hop = queue.popleft()

            if current == target and hop > 0:
                path_score = 1.0
                for eidx in path_edges:
                    if eidx < len(avg_attention):
                        path_score *= max(float(avg_attention[eidx]), 1e-10)
                    else:
                        path_score *= 0.001

                found_paths.append({
                    'nodes': list(path_nodes),
                    'edges': list(path_edges),
                    'relations': list(path_rels),
                    'score': max(path_score, 1e-50),
                    'hops': hop
                })
                continue

            if hop >= max_hop:
                continue

            for neighbor, relation, edge_idx in self.adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((
                        neighbor,
                        path_nodes + [neighbor],
                        path_edges + [edge_idx],
                        path_rels + [relation],
                        hop + 1
                    ))

        return found_paths

    def _diverse_rerank(self, paths, top_k):
        if len(paths) <= top_k:
            return sorted(paths, key=lambda p: p['score'], reverse=True)

        paths = sorted(paths, key=lambda p: p['score'], reverse=True)

        selected = []
        used_intermediates = set()
        used_rel_patterns = set()

        for p in paths:
            if len(selected) >= top_k:
                break

            intermediates = set(p['nodes'][1:-1])
            rel_pattern = tuple(p['relations'])

            if used_intermediates:
                overlap = len(intermediates & used_intermediates)
                overlap_ratio = overlap / max(len(intermediates), 1)
            else:
                overlap_ratio = 0.0

            pattern_penalty = 0.3 if rel_pattern in used_rel_patterns else 0.0
            diversity_penalty = overlap_ratio * 0.5 + pattern_penalty
            adjusted_score = p['score'] * (1.0 - diversity_penalty)

            if not selected or adjusted_score > 0:
                selected.append(p)
                used_intermediates.update(intermediates)
                used_rel_patterns.add(rel_pattern)

        return selected

    def _format_path(self, path_info):
        nodes = path_info['nodes']
        relations = path_info['relations']
        edges = path_info['edges']

        node_names = []
        for node_id in nodes:
            if node_id < self.n_users:
                node_names.append(self.dataset.get_part_name(node_id))
            else:
                entity_id = node_id - self.n_users
                node_names.append(self.dataset.get_entity_name(entity_id))

        rel_names = [self.dataset.get_relation_name(r) for r in relations]

        path_str_parts = [str(node_names[0])]
        for i, rel in enumerate(rel_names):
            path_str_parts.append(f" --[{rel}]--> ")
            path_str_parts.append(str(node_names[i + 1]))
        path_str = ''.join(path_str_parts)

        path_type = self._classify_path_type(nodes, relations)

        constraint_edges_in_path = []
        for eidx in edges:
            if eidx in self.constraint_edge_types:
                constraint_edges_in_path.append(self.constraint_edge_types[eidx])
            elif eidx in self.constraint_influenced_types:
                constraint_edges_in_path.append(self.constraint_influenced_types[eidx])

        modulation = self._get_attention_modulation()
        path_cam = 0.0
        edge_modulations = []

        if modulation is not None:
            attn_w = modulation['attn_with']
            attn_wo = modulation['attn_without']

            score_with = 1.0
            score_without = 1.0

            for i, eidx in enumerate(edges):
                if eidx < len(attn_w):
                    aw = max(float(attn_w[eidx]), 1e-20)
                    awo = max(float(attn_wo[eidx]), 1e-20)
                    score_with *= aw
                    score_without *= awo

                    rel_name = rel_names[i] if i < len(rel_names) else '?'
                    edge_diff = float(attn_w[eidx] - attn_wo[eidx])
                    edge_rel_mod = abs(edge_diff) / (abs(float(attn_wo[eidx])) + 1e-10)
                    edge_modulations.append({
                        'edge_idx': eidx,
                        'rel_name': rel_name,
                        'diff': edge_diff,
                        'rel_mod': edge_rel_mod,
                    })

            if score_without > 1e-50:
                path_cam = abs(score_with - score_without) / score_without

        return {
            'node_ids': nodes,
            'node_names': node_names,
            'relation_ids': relations,
            'relation_names': rel_names,
            'attention_score': float(path_info['score']),
            'hops': int(path_info['hops']),
            'path_string': path_str,
            'path_type': path_type,
            'constraint_edges': constraint_edges_in_path,
            'edge_modulations': edge_modulations,
            'constraint_influence': path_cam,
        }

    def _classify_path_type(self, nodes, relations):
        n_relations = self.dataset.n_relations
        interact_rel = n_relations * 2
        has_interact = interact_rel in relations
        has_kg = any(r < n_relations * 2 for r in relations)

        if has_interact and has_kg:
            return "hybrid"
        elif has_interact:
            return "collaborative"
        elif has_kg:
            return "knowledge"
        return "unknown"

    def _analyze_path_constraints(self, paths):
        total_edges = 0
        direct_edges = 0
        influenced_edges = 0
        type_counts = defaultdict(int)

        for p in paths:
            for eidx in p['edges']:
                total_edges += 1
                if eidx in self.constraint_edge_types:
                    direct_edges += 1
                    ctype = self.constraint_edge_types[eidx]['type_name']
                    type_counts[ctype] += 1
                elif eidx in self.constraint_influenced_types:
                    influenced_edges += 1
                    ctype = self.constraint_influenced_types[eidx]['type_name']
                    type_counts[ctype] += 1

        constraint_edges = direct_edges + influenced_edges

        return {
            'total_edges_in_paths': total_edges,
            'constraint_edges': constraint_edges,
            'direct_edges': direct_edges,
            'influenced_edges': influenced_edges,
            'activation_rate': (constraint_edges / total_edges if total_edges > 0 else 0),
            'type_distribution': dict(type_counts),
        }

    def _get_gate_values_for_recommendation(self, part_id, plan_id, paths):
        if not hasattr(self.model, 'gat_layers'):
            return {}

        gate_values = {}

        for layer_idx, layer in enumerate(self.model.gat_layers):
            gate_layer = None

            if hasattr(layer, 'constraint_gate'):
                gate_layer = layer
            elif hasattr(layer, 'heads'):
                for head in layer.heads:
                    if hasattr(head, 'constraint_gate'):
                        gate_layer = head
                        break

            if gate_layer is not None and hasattr(gate_layer, 'lambda_c'):
                if hasattr(gate_layer, 'get_bounded_lambdas'):
                    lc_bounded, lg_bounded = gate_layer.get_bounded_lambdas()
                    lc = lc_bounded.detach().cpu().numpy()
                    lg = lg_bounded.detach().cpu().item()
                else:
                    lc = gate_layer.lambda_c.detach().cpu().numpy()
                    lg = gate_layer.lambda_global.detach().cpu().item()

                gate_values[f'layer_{layer_idx}'] = {
                    'lambda_global': float(lg),
                    'lambda_per_type': {
                        CONSTRAINT_TYPES[i]: float(lc[i])
                        for i in range(min(len(lc), len(CONSTRAINT_TYPES)))
                    }
                }

        return gate_values

    def compute_explainability_metrics(self, test_pairs, top_k=EXPLAIN_TOP_K_PATHS):
        n_total = len(test_pairs)
        n_with_paths = 0
        all_scores = []
        all_lengths = []
        type_counts = defaultdict(int)

        total_path_edges = 0
        total_constraint_edges = 0
        total_direct_edges = 0
        total_influenced_edges = 0
        constraint_type_counts = defaultdict(int)

        concentration_values = []
        all_gate_values = defaultdict(list)
        all_path_cams = []

        print(f"\n   Evaluating explainability on {n_total} pairs...")

        for idx, (part_id, plan_id) in enumerate(test_pairs):
            if (idx + 1) % 10 == 0:
                print(f"   ... {idx + 1}/{n_total}")
            try:
                exp = self.explain_recommendation(part_id, plan_id, top_k)

                if exp['paths']:
                    n_with_paths += 1

                    path_scores = []
                    for p in exp['paths']:
                        path_scores.append(p['attention_score'])
                        all_scores.append(p['attention_score'])
                        all_lengths.append(p['hops'])
                        type_counts[p['path_type']] += 1

                        for ce in p.get('constraint_edges', []):
                            total_constraint_edges += 1
                            constraint_type_counts[ce['type_name']] += 1
                            if ce.get('match_level') == 'direct':
                                total_direct_edges += 1
                            else:
                                total_influenced_edges += 1

                    if len(path_scores) >= 2:
                        total_att = sum(path_scores)
                        if total_att > 0:
                            top3_att = sum(sorted(path_scores, reverse=True)[:3])
                            concentration_values.append(top3_att / total_att)

                    for p in exp['paths']:
                        total_path_edges += p['hops']

                    for p in exp['paths']:
                        cam = p.get('constraint_influence', 0.0)
                        if cam > 0:
                            all_path_cams.append(cam)

                if exp.get('gate_info'):
                    for _, ginfo in exp['gate_info'].items():
                        for ctype, val in ginfo.get('lambda_per_type', {}).items():
                            all_gate_values[ctype].append(val)

            except Exception:
                continue

        rsm_metrics = self.compute_rsm_metrics(test_pairs)

        metrics = {
            'path_coverage': n_with_paths / n_total if n_total > 0 else 0,
            'avg_attention_score': float(np.mean(all_scores)) if all_scores else 0,
            'avg_path_length': float(np.mean(all_lengths)) if all_lengths else 0,
            'total_paths_found': len(all_scores),
            'path_type_distribution': dict(type_counts),
            'attention_concentration': float(np.mean(concentration_values)) if concentration_values else 0,
            'constraint_activation_rate': (
                total_constraint_edges / total_path_edges if total_path_edges > 0 else 0
            ),
            'constraint_direct_rate': (
                total_direct_edges / total_path_edges if total_path_edges > 0 else 0
            ),
            'constraint_influenced_rate': (
                total_influenced_edges / total_path_edges if total_path_edges > 0 else 0
            ),
            'constraint_type_distribution': dict(constraint_type_counts),
            'gate_values': {
                ctype: {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                }
                for ctype, vals in all_gate_values.items()
                if vals
            },
            'cam_mean': float(np.mean(all_path_cams)) if all_path_cams else 0.0,
            'cam_std': float(np.std(all_path_cams)) if all_path_cams else 0.0,
            'cam_paths_with_modulation': len(all_path_cams),
            'constrained_edge_cam': (self._get_attention_modulation() or {}).get('constrained_cam', 0.0),
            'global_edge_cam': (self._get_attention_modulation() or {}).get('global_cam', 0.0),
            'rsm_mean': rsm_metrics['rsm_mean'],
            'rsm_std': rsm_metrics['rsm_std'],
            'rank_change_mean': rsm_metrics['rank_change_mean'],
            'rank_degraded_ratio': rsm_metrics['rank_degraded_ratio'],
            'n_evaluated': n_total,
            'n_with_paths': n_with_paths,
        }

        return metrics

    def compute_global_cam(self):
        mod = self._get_attention_modulation()
        if mod is None:
            return {
                'global_cam': 0.0,
                'kg_edge_cam': 0.0,
                'constrained_cam': 0.0,
                'n_modulated': 0,
                'n_total': 0
            }

        return {
            'global_cam': mod['global_cam'],
            'kg_edge_cam': mod['kg_edge_cam'],
            'constrained_cam': mod['constrained_cam'],
            'n_modulated': int((mod['abs_diff'] > 1e-6).sum()),
            'n_total': len(mod['diff']),
        }

    def print_explanation(self, exp):
        print(f"\n   {'─' * 60}")
        print(f"   Part: {exp['part']} (ID={exp['part_id']})")
        print(f"   Plan: {exp['plan']} (ID={exp['plan_id']})")
        rsm = exp.get('rsm', 0)
        s_wo = exp.get('score_without_constraint', 0)
        print(f"   Score: {exp['score']:.4f} (w/o constraint: {s_wo:.4f}, RSM: {rsm:.2%})")
        print(f"   {'─' * 60}")

        if not exp['paths']:
            print("   No explainable paths found")
            return

        for i, p in enumerate(exp['paths']):
            print(
                f"   Path {i + 1} "
                f"(score={p['attention_score']:.6f}, hops={p['hops']}, type={p['path_type']}):"
            )
            print(f"     {p['path_string']}")

            if p.get('constraint_edges'):
                for ce in p['constraint_edges']:
                    level = ce.get('match_level', 'direct')
                    if level == 'direct':
                        print(
                            f"     Constraint (direct): {ce['type_name']} "
                            f"(value={ce.get('value', 0):.2f})"
                        )
                    else:
                        print(f"     Constraint (influenced): {ce['type_name']}")

        if exp.get('gate_info'):
            print("\n   Gate Values:")
            for layer_key, ginfo in exp['gate_info'].items():
                print(f"     {layer_key}: lambda_global={ginfo['lambda_global']:.4f}")
                for ctype, val in ginfo.get('lambda_per_type', {}).items():
                    print(f"       lambda_{ctype}: {val:.4f}")

    def print_metrics(self, metrics):
        print(f"\n   {'=' * 60}")
        print("   Explainability Metrics")
        print(f"   {'=' * 60}")
        print(
            f"   Path Coverage:           {metrics['path_coverage']:.4f} "
            f"({metrics['n_with_paths']}/{metrics['n_evaluated']})"
        )
        print(f"   Avg Attention Score:     {metrics['avg_attention_score']:.6f}")
        print(f"   Avg Path Length:         {metrics['avg_path_length']:.2f} hops")
        print(f"   Attention Concentration: {metrics['attention_concentration']:.4f}")
        print(
            f"   Constraint Activation:   {metrics['constraint_activation_rate']:.4f} "
            f"(direct={metrics.get('constraint_direct_rate', 0):.4f}, "
            f"influenced={metrics.get('constraint_influenced_rate', 0):.4f})"
        )
        print(f"   Constrained-Edge CAM:    {metrics.get('constrained_edge_cam', 0):.4f}")
        print(
            f"   Path-Score CAM:          {metrics.get('cam_mean', 0):.4f} "
            f"+/- {metrics.get('cam_std', 0):.4f} "
            f"({metrics.get('cam_paths_with_modulation', 0)} paths)"
        )
        print(
            f"   Score Modulation:        {metrics.get('rsm_mean', 0):.4f} "
            f"+/- {metrics.get('rsm_std', 0):.4f}"
        )
        print(
            f"   Rank Degraded Ratio:     {metrics.get('rank_degraded_ratio', 0):.2%} "
            f"(avg change: {metrics.get('rank_change_mean', 0):+.1f})"
        )
        print(f"   Total Paths Found:       {metrics['total_paths_found']}")

        print("\n   Path Type Distribution:")
        for ptype, count in metrics['path_type_distribution'].items():
            pct = count / metrics['total_paths_found'] * 100 if metrics['total_paths_found'] > 0 else 0
            print(f"     {ptype:15s}: {count:4d} ({pct:.1f}%)")

        if metrics['constraint_type_distribution']:
            print("\n   Constraint Type Activation:")
            for ctype, count in metrics['constraint_type_distribution'].items():
                print(f"     {ctype:25s}: {count:4d}")

        if metrics['gate_values']:
            print("\n   Learned Gate Values:")
            for ctype, stats in metrics['gate_values'].items():
                print(f"     {ctype:25s}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

        print(f"   {'=' * 60}")