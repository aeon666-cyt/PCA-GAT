import torch
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict

from config_pcagat import *
from dataset_machining import load_dataset
from model_pcagat import PCAGAT
from explainer import PCAGATExplainer
from utils_pcagat import evaluate_model


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_trained_model(dataset, checkpoint_path=None, dim=64):
    edge_index, edge_type = dataset.get_edge_data()
    c_indices, c_values, c_types = dataset.build_full_constraint_matrix(
        n_nodes=dataset.n_users + dataset.n_entities)

    model = PCAGAT(
        n_users=dataset.n_users,
        n_entities=dataset.n_entities,
        n_relations=dataset.n_relations,
        n_items=dataset.n_items,
        dim=dim,
        item2entity=dataset.item2entity,
        edge_index=edge_index.to(DEVICE),
        edge_type=edge_type.to(DEVICE),
        constraint_indices=c_indices.to(DEVICE),
        constraint_values=c_values.to(DEVICE),
        constraint_types=c_types.to(DEVICE),
        n_layers=1, n_heads=1, dropout=0.1,
        use_constraint=True, use_kg=True,
        use_multi_head=False, use_gate=True,
        device=DEVICE
    ).to(DEVICE)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"   Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE,
                          weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"   ✓ Loaded successfully")
    else:
        print(f"   ⚠ No checkpoint found, training from scratch...")
        from utils_pcagat import train_pcagat
        model, _ = train_pcagat(
            model, dataset, epochs=300, lr=0.001,
            contrastive_weight=0, use_hard_neg=False,
            patience=15, min_epochs=30, eval_interval=5,
            checkpoint_path=checkpoint_path, verbose=True)

    model.eval()
    return model


def collect_test_pairs(dataset, max_pairs=None):
    pairs = []
    for part_id, plan_ids in dataset.test_dict.items():
        for plan_id in plan_ids:
            pairs.append((part_id, plan_id))
    if max_pairs and len(pairs) > max_pairs:
        np.random.shuffle(pairs)
        pairs = pairs[:max_pairs]
    return pairs


def run_case_studies(explainer, dataset, n_cases=5):
    print(f"\n{'='*70}")
    print(f"   Phase 1: Case Studies ({n_cases} cases)")
    print(f"{'='*70}")

    selected_parts = []
    for part_id in dataset.test_dict.keys():
        if len(dataset.test_dict[part_id]) > 0:
            selected_parts.append(part_id)
        if len(selected_parts) >= n_cases:
            break

    cases = []
    for part_id in selected_parts:
        plan_id = dataset.test_dict[part_id][0]

        exp = explainer.explain_recommendation(part_id, plan_id, top_k=3)
        explainer.print_explanation(exp)
        cases.append(exp)

    return cases


def run_quantitative_metrics(explainer, test_pairs):
    print(f"\n{'='*70}")
    print(f"   Phase 2: Quantitative Explainability Metrics")
    print(f"{'='*70}")

    metrics = explainer.compute_explainability_metrics(test_pairs, top_k=3)
    explainer.print_metrics(metrics)

    return metrics


def run_gate_analysis(model, dataset):
    print(f"\n{'='*70}")
    print(f"   Phase 3: Constraint Gate Value Analysis")
    print(f"{'='*70}")

    constraint_weights = model.get_constraint_weights()

    print("\n   Global Learned Constraint Weights:")
    for layer_key, weights in constraint_weights.items():
        print(f"\n   {layer_key}:")
        if 'lambda_global' in weights:
            lg = weights['lambda_global']
            print(f"     λ_global: {lg:.4f}"
                  if isinstance(lg, float)
                  else f"     λ_global: {lg}")
        if 'lambda_c' in weights:
            lc = weights['lambda_c']
            if hasattr(lc, '__len__'):
                for i, ctype in enumerate(CONSTRAINT_TYPES):
                    if i < len(lc):
                        print(f"     λ_{ctype}: {lc[i]:.4f}")
        if weights.get('has_gate'):
            print(f"     Gate: Active ✓")

    print("\n   Per-Part Gate Activation Patterns:")

    part_gate_data = {}
    for part_id in list(dataset.test_dict.keys())[:20]:
        plan_ids = dataset.test_dict[part_id]
        if not plan_ids:
            continue
        plan_id = plan_ids[0]

        part_name = dataset.get_part_name(part_id)
        part_entities = set()
        for item in dataset.train_set.get(part_id, []):
            ent_id = dataset.item2entity.get(item)
            if ent_id is not None:
                for r, t in dataset.kg_dict.get(ent_id, []):
                    entity_name = dataset.get_entity_name(t)
                    part_entities.add(entity_name)

        part_gate_data[part_id] = {
            'name': part_name,
            'related_entities': list(part_entities)[:5],
        }

    return {
        'global_weights': constraint_weights,
        'per_part_data': part_gate_data,
    }


def run_attention_distribution(explainer, test_pairs):
    print(f"\n{'='*70}")
    print(f"   Phase 4: Attention Distribution Analysis")
    print(f"{'='*70}")

    type_scores = defaultdict(list)
    hop_scores = defaultdict(list)

    for part_id, plan_id in test_pairs[:50]:
        try:
            exp = explainer.explain_recommendation(part_id, plan_id, top_k=5)
            for p in exp['paths']:
                type_scores[p['path_type']].append(p['attention_score'])
                hop_scores[p['hops']].append(p['attention_score'])
        except:
            continue

    print("\n   Attention Score by Path Type:")
    print(f"   {'Type':<20} {'Count':>6} {'Mean':>12} {'Std':>12}")
    print(f"   {'-'*55}")
    for ptype in ['knowledge', 'collaborative', 'hybrid']:
        scores = type_scores.get(ptype, [])
        if scores:
            print(f"   {ptype:<20} {len(scores):>6} "
                  f"{np.mean(scores):>12.6f} {np.std(scores):>12.6f}")

    print("\n   Attention Score by Path Length:")
    print(f"   {'Hops':<10} {'Count':>6} {'Mean':>12} {'Std':>12}")
    print(f"   {'-'*45}")
    for hop in sorted(hop_scores.keys()):
        scores = hop_scores[hop]
        print(f"   {hop:<10} {len(scores):>6} "
              f"{np.mean(scores):>12.6f} {np.std(scores):>12.6f}")

    return {
        'by_type': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                        'count': len(v)}
                    for k, v in type_scores.items()},
        'by_hops': {str(k): {'mean': float(np.mean(v)),
                             'std': float(np.std(v)), 'count': len(v)}
                    for k, v in hop_scores.items()},
    }


def main():
    set_seed(42)

    print(f"\n{'='*70}")
    print(f"   RQ3: PCA-GAT Explainability Analysis")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    dataset = load_dataset()
    model = load_trained_model(dataset, PCAGAT_CHECKPOINT, dim=128)

    print("\n   Quick performance check:")
    with torch.no_grad():
        metrics = evaluate_model(model, dataset, top_k_list=[1, 5])
    print(f"   R@1={metrics[1]['Recall']:.4f}, R@5={metrics[5]['Recall']:.4f}")

    explainer = PCAGATExplainer(model, dataset)
    test_pairs = collect_test_pairs(dataset)
    print(f"\n   Total test pairs: {len(test_pairs)}")

    cases = run_case_studies(explainer, dataset, n_cases=5)
    metrics = run_quantitative_metrics(explainer, test_pairs)
    gate_data = run_gate_analysis(model, dataset)
    attn_data = run_attention_distribution(explainer, test_pairs)

    cam_data = explainer.compute_global_cam()
    print(f"\n   ★ Global CAM: {cam_data['global_cam']:.4f}")
    print(f"   ★ KG-edge CAM: {cam_data['kg_edge_cam']:.4f}")
    print(f"   ★ Constrained-edge CAM: {cam_data['constrained_cam']:.4f}")
    print(f"   ★ Modulated edges: {cam_data['n_modulated']}/{cam_data['n_total']}")

    rsm_data = explainer.compute_rsm_metrics(test_pairs)
    print(f"\n   ★ RSM (Score Modulation): {rsm_data['rsm_mean']:.4f} ± {rsm_data['rsm_std']:.4f}")
    print(f"   ★ Rank degraded w/o constraint: "
          f"{rsm_data['rank_degraded']}/{rsm_data['n_pairs']} "
          f"({rsm_data['rank_degraded_ratio']:.2%})")
    print(f"   ★ Avg rank change: {rsm_data['rank_change_mean']:+.2f}")

    save_path = os.path.join(RESULT_DIR, 'rq3_explainability.json')

    serializable_cases = []
    for exp in cases:
        case = {
            'part': exp['part'],
            'part_id': exp['part_id'],
            'plan': exp['plan'],
            'plan_id': exp['plan_id'],
            'score': exp['score'],
            'constraint_info': exp['constraint_info'],
            'paths': []
        }
        for p in exp['paths']:
            case['paths'].append({
                'path_string': p['path_string'],
                'attention_score': p['attention_score'],
                'hops': p['hops'],
                'path_type': p['path_type'],
                'constraint_edges': p.get('constraint_edges', []),
                'node_names': p['node_names'],
                'relation_names': p['relation_names'],
            })
        serializable_cases.append(case)

    serializable_gate = {}
    for k, v in gate_data.get('global_weights', {}).items():
        sg = {}
        if 'lambda_c' in v:
            sg['lambda_c'] = (v['lambda_c'].tolist()
                              if hasattr(v['lambda_c'], 'tolist')
                              else list(v['lambda_c']))
        if 'lambda_global' in v:
            sg['lambda_global'] = float(v['lambda_global'])
        sg['has_gate'] = v.get('has_gate', False)
        serializable_gate[k] = sg

    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'case_studies': serializable_cases,
        'metrics': {
            k: (float(v) if isinstance(v, (int, float, np.floating))
                else v)
            for k, v in metrics.items()
        },
        'gate_analysis': serializable_gate,
        'attention_distribution': attn_data,
        'cam': cam_data,
        'rsm': rsm_data,
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n   ✓ Full results saved to {save_path}")

    print(f"\n\n{'='*70}")
    print("   📋 PAPER-READY OUTPUTS")
    print(f"{'='*70}")

    print("\n   Table X: Explainability Metrics of PCA-GAT")
    print(f"   | Metric | PCA-GAT | KGAT |")
    print(f"   |---|---|---|")
    print(f"   | Path Coverage | {metrics['path_coverage']:.2%} | - |")
    print(f"   | Avg Path Length | {metrics['avg_path_length']:.2f} | - |")
    print(f"   | Attention Concentration | "
          f"{metrics['attention_concentration']:.4f} | - |")
    print(f"   | Constraint Activation Rate | "
          f"{metrics['constraint_activation_rate']:.2%} | 0.00% |")
    print(f"   | ★ Score Modulation (RSM) | "
          f"{metrics.get('rsm_mean', 0):.4f} | 0.0000 |")
    print(f"   | ★ Rank Degraded w/o Constraint | "
          f"{metrics.get('rank_degraded_ratio', 0):.2%} | 0.00% |")
    print(f"   | Constrained-Edge CAM | "
          f"{cam_data.get('constrained_cam', 0):.4f} | 0.0000 |")

    total_paths = metrics['total_paths_found']
    print(f"\n   | Path Type | Count | Ratio |")
    print(f"   |---|---|---|")
    for ptype, count in sorted(metrics['path_type_distribution'].items(),
                               key=lambda x: -x[1]):
        print(f"   | {ptype} | {count} | "
              f"{count/total_paths:.2%} |")

    if metrics['gate_values']:
        print(f"\n   Table Y: Learned Constraint Weights")
        print(f"   | Constraint Type | λ (mean±std) |")
        print(f"   |---|---|")
        for ctype, stats in metrics['gate_values'].items():
            print(f"   | {ctype} | "
                  f"{stats['mean']:.4f}±{stats['std']:.4f} |")

    print(f"\n{'='*70}")
    print(f"   ✅ RQ3 Analysis Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()