import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import json
import os
import numpy as np
from datetime import datetime

from config_pcagat import (
    DEVICE, PCAGAT_EPOCHS, PCAGAT_LR, PCAGAT_KG_WEIGHT,
    PCAGAT_L2_WEIGHT, PCAGAT_PATIENCE, PCAGAT_MIN_EPOCHS,
    PCAGAT_EVAL_INTERVAL, PCAGAT_CONSTRAINT_LR_MULT,
    USE_CONSTRAINT_GATE, RESULT_DIR,
    USE_CONSTRAINT_ALIGNMENT, CONSTRAINT_ALIGNMENT_WEIGHT,
    CONSTRAINT_ALIGNMENT_SAMPLE, SEED,
)
from dataset_machining import load_dataset
from model_pcagat import PCAGAT
from utils_pcagat import (
    evaluate_model, compute_constraint_satisfaction, print_results
)

DIMS = [48, 64, 96, 128]
LAYERS = [1, 2]
HEADS = [1, 2]
DROPOUTS = [0.10, 0.15, 0.20, 0.25, 0.30]

RANK_BY = 'score'

def train_single(model, dataset):
    constraint_params, other_params = [], []
    for name, param in model.named_parameters():
        if 'lambda_c' in name or 'lambda_global' in name or 'constraint_gate' in name:
            constraint_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.Adam([
        {'params': other_params, 'lr': PCAGAT_LR},
        {'params': constraint_params, 'lr': PCAGAT_LR * PCAGAT_CONSTRAINT_LR_MULT},
    ])

    inter_loader = dataset.get_interaction_loader(use_hard_neg=False)
    kg_loader = dataset.get_kg_loader() if model.use_kg else None

    alignment_weight = CONSTRAINT_ALIGNMENT_WEIGHT if USE_CONSTRAINT_ALIGNMENT else 0.0
    use_alignment = (alignment_weight > 0
                     and model.use_constraint
                     and model.constraint_indices.size(1) > 0)

    best_r1, patience_cnt, best_state = 0.0, 0, None

    for epoch in range(PCAGAT_EPOCHS):
        model.train()
        total_loss, n_bat = 0, 0
        kg_iter = iter(kg_loader) if kg_loader else None

        for batch in inter_loader:
            batch = batch.to(DEVICE)
            u, pi, ni = batch[:, 0], batch[:, 1], batch[:, 2]
            optimizer.zero_grad()
            ps, ns = model(u, pi, ni)
            bpr = -torch.mean(F.logsigmoid(ps - ns))

            kg_l = torch.tensor(0.0, device=DEVICE)
            if kg_loader and PCAGAT_KG_WEIGHT > 0:
                try:
                    kb = next(kg_iter)
                except StopIteration:
                    kg_iter = iter(kg_loader)
                    kb = next(kg_iter)
                kb = kb.to(DEVICE)
                kg_l = model.compute_kg_loss(kb[:, 0], kb[:, 1], kb[:, 2], kb[:, 3])

            l2 = model.compute_l2_loss(u, pi, ni)

            align_loss = (model.compute_constraint_alignment_loss(
                              sample_size=CONSTRAINT_ALIGNMENT_SAMPLE)
                          if use_alignment
                          else torch.tensor(0.0, device=DEVICE))

            loss = (bpr
                    + PCAGAT_KG_WEIGHT * kg_l
                    + PCAGAT_L2_WEIGHT * l2
                    + alignment_weight * align_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n_bat += 1

        if epoch + 1 >= PCAGAT_MIN_EPOCHS and (epoch + 1) % PCAGAT_EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                m = evaluate_model(model, dataset, top_k_list=[1, 5])
                r1 = m[1]['Recall']
            if r1 > best_r1 + 1e-4:
                best_r1 = r1
                patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
            if patience_cnt >= PCAGAT_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model

def main():
    output_path = os.path.join(RESULT_DIR, 'grid_search_results.json')
    top_path = os.path.join(RESULT_DIR, 'grid_search_top20.json')
    os.makedirs(RESULT_DIR, exist_ok=True)

    if PCAGAT_L2_WEIGHT > 1e-5:
        print(f"\n  ⚠️  WARNING: PCAGAT_L2_WEIGHT={PCAGAT_L2_WEIGHT} 看起来太大！")
        print(f"  ⚠️  请先运行 python fix_regression.py")
        print(f"  ⚠️  继续运行但结果可能不理想\n")

    configs = []
    for d in DIMS:
        for L in LAYERS:
            for H in HEADS:
                if d % H != 0:
                    continue
                for drop in DROPOUTS:
                    name = f"d{d}_L{L}_H{H}_dr{drop:.2f}"
                    configs.append((name, d, L, H, drop))

    total = len(configs)
    print(f"\n{'='*80}")
    print(f"   PCA-GAT Grid Search — {total} configurations")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Device: {DEVICE}")
    print(f"   L2 weight: {PCAGAT_L2_WEIGHT}")
    print(f"   Search: d={DIMS}, L={LAYERS}, H={HEADS}, drop={DROPOUTS}")
    print(f"{'='*80}\n")

    dataset = load_dataset()
    edge_index, edge_type = dataset.get_edge_data()
    c_idx, c_val, c_typ = dataset.build_full_constraint_matrix(
        n_nodes=dataset.n_users + dataset.n_entities)

    all_results = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            all_results = json.load(f)
        print(f"   Loaded {len(all_results)} existing results (skip)\n")

    total_start = time.time()

    for i, (name, d, L, H, dropout) in enumerate(configs):
        if name in all_results:
            print(f"   [{i+1}/{total}] {name} — SKIP")
            continue

        print(f"\n{'─'*70}")
        print(f"   [{i+1}/{total}] {name}  (d={d}, L={L}, H={H}, drop={dropout})")
        print(f"{'─'*70}")

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        t0 = time.time()

        model = PCAGAT(
            n_users=dataset.n_users, n_entities=dataset.n_entities,
            n_relations=dataset.n_relations, n_items=dataset.n_items,
            dim=d, item2entity=dataset.item2entity,
            edge_index=edge_index.to(DEVICE), edge_type=edge_type.to(DEVICE),
            constraint_indices=c_idx.to(DEVICE),
            constraint_values=c_val.to(DEVICE),
            constraint_types=c_typ.to(DEVICE),
            n_layers=L, n_heads=H, dropout=dropout,
            use_constraint=True, use_kg=True,
            use_multi_head=(H > 1), use_gate=USE_CONSTRAINT_GATE,
            device=DEVICE
        ).to(DEVICE)

        n_p = sum(p.numel() for p in model.parameters())
        mb = n_p * 4 / (1024 * 1024)

        model = train_single(model, dataset)
        train_time = time.time() - t0

        model.eval()
        metrics = evaluate_model(model, dataset, top_k_list=[1, 3, 5, 10, 20])
        csr5 = compute_constraint_satisfaction(model, dataset, top_k=5)

        r = {
            'name': name, 'd': d, 'L': L, 'H': H, 'dropout': dropout,
            'n_params': n_p, 'n_params_mb': round(mb, 3),
            'train_time_s': round(train_time, 1),
            'train_time_min': round(train_time / 60, 1),
        }
        for k in [1, 3, 5, 10, 20]:
            r[f'R{k}']    = round(metrics[k]['Recall'], 4)
            r[f'P{k}']    = round(metrics[k]['Precision'], 4)
            r[f'HR{k}']   = round(metrics[k]['HR'], 4)
            r[f'NDCG{k}'] = round(metrics[k]['NDCG'], 4)
            r[f'MRR{k}']  = round(metrics[k]['MRR'], 4)
        r['CSR5'] = round(csr5['csr'], 4)
        r['score'] = round(
            0.4 * r['R1'] + 0.3 * r['R5'] + 0.2 * r['MRR5'] + 0.1 * r['NDCG5'], 4)

        all_results[name] = r

        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"   ✓ R@1={r['R1']:.4f}  R@5={r['R5']:.4f}  "
              f"MRR@5={r['MRR5']:.4f}  score={r['score']:.4f}  "
              f"| {mb:.3f}MB  {r['train_time_min']:.1f}min")
        print(f"   ✓ Progress: {len(all_results)}/{total}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    sorted_results = sorted(all_results.values(),
                            key=lambda x: x.get(RANK_BY, x.get('score', 0)),
                            reverse=True)

    with open(top_path, 'w') as f:
        json.dump(sorted_results[:20], f, indent=2, ensure_ascii=False)

    print(f"\n\n{'='*110}")
    print(f"   GRID SEARCH COMPLETE — {len(all_results)} configs, "
          f"Total: {total_time/60:.1f} min")
    print(f"{'='*110}")
    print(f"  {'Rank':>4} {'Name':>22} | {'d':>3} {'L':>2} {'H':>2} {'drop':>5} | "
          f"{'R@1':>7} {'R@5':>7} {'MRR@5':>7} {'NDCG@5':>7} {'CSR@5':>6} | "
          f"{'Score':>7} | {'MB':>6} {'Time':>6}")
    print(f"  {'─'*106}")

    for rank, r in enumerate(sorted_results, 1):
        marker = ' ★' if rank <= 3 else '  '
        print(f"{marker}{rank:>3}. {r['name']:>22} | "
              f"{r['d']:>3} {r['L']:>2} {r['H']:>2} {r['dropout']:>5.2f} | "
              f"{r['R1']:>7.4f} {r['R5']:>7.4f} {r['MRR5']:>7.4f} "
              f"{r['NDCG5']:>7.4f} {r['CSR5']:>6.4f} | "
              f"{r['score']:>7.4f} | "
              f"{r['n_params_mb']:>5.3f} {r['train_time_min']:>5.1f}m")

    print(f"{'='*110}")

    print(f"\n   📊 Best per dimension:")
    for d in DIMS:
        d_results = [r for r in sorted_results if r['d'] == d]
        if d_results:
            b = d_results[0]
            print(f"      d={d:>3}: {b['name']:>22}  R@1={b['R1']:.4f}  R@5={b['R5']:.4f}  score={b['score']:.4f}")

    print(f"\n   📊 Best per layer:")
    for L in LAYERS:
        l_results = [r for r in sorted_results if r['L'] == L]
        if l_results:
            b = l_results[0]
            print(f"      L={L}: {b['name']:>22}  R@1={b['R1']:.4f}  R@5={b['R5']:.4f}  score={b['score']:.4f}")

    best = sorted_results[0]
    print(f"\n   🏆 BEST: {best['name']}")
    print(f"      d={best['d']}, L={best['L']}, H={best['H']}, drop={best['dropout']}")
    print(f"      R@1={best['R1']:.4f}  R@5={best['R5']:.4f}  "
          f"MRR@5={best['MRR5']:.4f}  NDCG@5={best['NDCG5']:.4f}")
    print(f"      {best['n_params_mb']:.3f} MB  {best['train_time_min']:.1f} min\n")

if __name__ == '__main__':
    main()