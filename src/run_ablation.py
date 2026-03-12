import torch
import torch.nn.functional as F
import time
import json
import os
import numpy as np
from datetime import datetime
from collections import OrderedDict

from config_pcagat import *

TOP_K_LIST = [1, 3, 5]

from dataset_machining import load_dataset
from model_pcagat import PCAGAT
from utils_pcagat import (
    train_pcagat,
    evaluate_model, compute_constraint_satisfaction, print_results
)


def run_pcagat_variant(dataset, edge_index, edge_type,
                       c_indices, c_values, c_types,
                       variant_name, config):
    print(f"\n{'='*70}")
    print(f"   Training: {config['desc']}")
    print(f"{'='*70}")
    print(f"   Constraint={config['use_constraint']}, "
          f"Gate={config.get('use_gate', False)}, "
          f"KG={config['use_kg']}, "
          f"Layers={config['n_layers']}")

    if config['use_constraint']:
        ci = c_indices.to(DEVICE)
        cv = c_values.to(DEVICE)
        ct = c_types.to(DEVICE)
    else:
        ci = torch.zeros(2, 0, dtype=torch.long, device=DEVICE)
        cv = torch.zeros(0, device=DEVICE)
        ct = torch.zeros(0, dtype=torch.long, device=DEVICE)

    model = PCAGAT(
        n_users=dataset.n_users,
        n_entities=dataset.n_entities,
        n_relations=dataset.n_relations,
        n_items=dataset.n_items,
        dim=EMBED_DIM,
        item2entity=dataset.item2entity,
        edge_index=edge_index.to(DEVICE),
        edge_type=edge_type.to(DEVICE),
        constraint_indices=ci,
        constraint_values=cv,
        constraint_types=ct,
        n_layers=config['n_layers'],
        n_heads=1,
        dropout=PCAGAT_DROPOUT,
        use_constraint=config['use_constraint'],
        use_kg=config['use_kg'],
        use_multi_head=False,
        use_gate=config.get('use_gate', False),
        constraint_margin=CONSTRAINT_CONTRASTIVE_MARGIN,
        device=DEVICE
    ).to(DEVICE)

    checkpoint = os.path.join(CHECKPOINT_DIR, f'pcagat_{variant_name}.pth')
    kg_weight = PCAGAT_KG_WEIGHT if config['use_kg'] else 0

    start_time = time.time()
    model, history = train_pcagat(
        model, dataset,
        kg_weight=kg_weight,
        contrastive_weight=0,
        use_hard_neg=False,
        hard_neg_ratio=0,
        checkpoint_path=checkpoint
    )
    train_time = time.time() - start_time

    model.eval()
    results = evaluate_model(model, dataset, TOP_K_LIST)
    print_results(results, config['desc'])

    csr = compute_constraint_satisfaction(model, dataset, top_k=max(TOP_K_LIST))
    print(f"   CSR@{max(TOP_K_LIST)}: {csr['csr']:.4f}")

    return results, train_time, history, csr


def format_ablation_table(all_results):
    lines = []
    lines.append("")
    lines.append("=" * 110)
    lines.append("  Ablation Study Results")
    lines.append("=" * 110)

    header = "| Model | Recall@1 | Recall@3 | Recall@5 | NDCG@1 | NDCG@3 | NDCG@5 | CSR@5 |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['results'].get(5, {}).get('Recall', 0),
        reverse=True
    )

    for name, data in sorted_results:
        r1 = data['results'].get(1, {}).get('Recall', 0)
        r3 = data['results'].get(3, {}).get('Recall', 0)
        r5 = data['results'].get(5, {}).get('Recall', 0)
        n1 = data['results'].get(1, {}).get('NDCG', 0)
        n3 = data['results'].get(3, {}).get('NDCG', 0)
        n5 = data['results'].get(5, {}).get('NDCG', 0)
        csr_val = data.get('csr', {}).get('csr', 0)

        lines.append(
            f"| {name} | {r1:.4f} | {r3:.4f} | {r5:.4f} | "
            f"{n1:.4f} | {n3:.4f} | {n5:.4f} | {csr_val:.4f} |"
        )

    lines.append("=" * 110)
    return '\n'.join(lines)


def main():
    print("\n" + "=" * 70)
    print("   PCA-GAT ABLATION STUDY (V3.3 - Slim)")
    print("=" * 70)
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Device: {DEVICE}")
    print(f"   Variants: {len(ABLATION_CONFIGS)}")
    print("=" * 70 + "\n")

    total_start = time.time()

    print("Step 1: Loading dataset...")
    dataset = load_dataset()

    edge_index, edge_type = dataset.get_edge_data()
    c_indices, c_values, c_types = dataset.build_full_constraint_matrix(
        n_nodes=dataset.n_users + dataset.n_entities
    )

    all_results = OrderedDict()

    print(f"\n\n{'#'*70}")
    print(f"#  PCA-GAT ABLATION VARIANTS")
    print(f"{'#'*70}")

    for variant_name, config in ABLATION_CONFIGS.items():
        results, t, _, csr = run_pcagat_variant(
            dataset, edge_index, edge_type,
            c_indices, c_values, c_types,
            variant_name, config
        )
        all_results[config['desc']] = {
            'results': results, 'time': t, 'csr': csr}

    print(f"\n\n{'#'*70}")
    print(f"#  ABLATION STUDY SUMMARY")
    print(f"{'#'*70}")

    print(format_ablation_table(all_results))

    json_results = OrderedDict()
    for name, data in all_results.items():
        jr = {
            'results': {
                str(k): {m: float(v) for m, v in metrics.items()}
                for k, metrics in data['results'].items()
            },
            'time': float(data['time']),
        }
        if 'csr' in data and data['csr']:
            jr['csr'] = {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in data['csr'].items()}
        json_results[name] = jr

    json_path = os.path.join(RESULT_DIR, 'ablation_results_v3.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(RESULT_DIR, 'ablation_summary_v3.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"PCA-GAT V3.3 Ablation Study (Slim)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(format_ablation_table(all_results))
        f.write('\n\n')

        f.write(f"\n{'='*70}\n")
        f.write(f"  Component Contribution Analysis\n")
        f.write(f"{'='*70}\n\n")

        max_k = max(TOP_K_LIST)
        full_data = all_results.get('PCA-GAT (Ours)', {})
        full_recall = full_data.get('results', {}).get(max_k, {}).get('Recall', 0)

        if full_recall > 0:
            f.write(f"  Base: PCA-GAT (Ours) | Recall@{max_k}={full_recall:.4f}\n\n")
            for name, data in all_results.items():
                if name == 'PCA-GAT (Ours)':
                    continue
                v_recall = data['results'].get(max_k, {}).get('Recall', 0)
                if v_recall > 0:
                    change = (full_recall - v_recall) / full_recall * 100
                    f.write(f"  vs {name:<30} | Δ Recall: {change:+.2f}%\n")

    print(f"\n   Results saved to {json_path}")
    print(f"   Summary saved to {summary_path}")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"   ABLATION STUDY COMPLETE!")
    print(f"   Experiments: {len(all_results)}, Total Time: {total_time/60:.1f} min")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()