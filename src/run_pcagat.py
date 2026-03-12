import torch
import time
import json
import os
import numpy as np
from datetime import datetime
from config_pcagat import USE_AMAZON
from config_pcagat import *
from dataset_machining import load_dataset
from model_pcagat import PCAGAT
from utils_pcagat import (
    pretrain_bprmf, train_pcagat, evaluate_model,
    compute_constraint_satisfaction,
    print_results, save_results
)

ITEM_CL_WEIGHT      = 0.0
ITEM_CL_NEG_SAMPLES = 64
ITEM_CL_START_EPOCH = 10


def main():
    print("\n" + "=" * 70)
    print("   PCA-GAT: Process-Constraint-Aware Graph Attention Network")
    print("   V6.0 - Bi-Interaction Aggregator")
    print("=" * 70)
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Device: {DEVICE}")
    print(f"   ★ Bi-Interaction: {USE_BI_INTERACTION}")
    print("=" * 70 + "\n")

    start_time = time.time()

    print("Step 1: Loading Dataset...")
    dataset = load_dataset()

    print("\nBuilding CKG graph structure...")
    edge_index, edge_type = dataset.get_edge_data()

    print("\nBuilding process constraint matrix...")
    c_indices, c_values, c_types = dataset.build_full_constraint_matrix(
        n_nodes=dataset.n_users + dataset.n_entities
    )
    print(f"   Constraint entries in CKG space: {c_values.size(0)}")

    pretrain_user_emb, pretrain_item_emb = None, None

    if USE_PRETRAIN:
        if os.path.exists(PRETRAIN_CHECKPOINT):
            print(f"\nFound existing pretrain: {PRETRAIN_CHECKPOINT}")
            pretrain_data = torch.load(
                PRETRAIN_CHECKPOINT, map_location='cpu', weights_only=True)
            pretrain_user_emb = pretrain_data['user_emb']
            pretrain_item_emb = pretrain_data['item_emb']
            print(f"   Loaded (Recall@5: "
                  f"{pretrain_data.get('best_recall', 'N/A')})")
        else:
            pretrain_user_emb, pretrain_item_emb = pretrain_bprmf(dataset)

    print(f"\n{'='*70}")
    print("   Phase 1: Initializing PCA-GAT Model (V6.0)")
    print(f"{'='*70}")

    model = PCAGAT(
        n_users=dataset.n_users,
        n_entities=dataset.n_entities,
        n_relations=dataset.n_relations,
        n_items=dataset.n_items,
        dim=EMBED_DIM,
        item2entity=dataset.item2entity,
        edge_index=edge_index.to(DEVICE),
        edge_type=edge_type.to(DEVICE),
        constraint_indices=c_indices.to(DEVICE),
        constraint_values=c_values.to(DEVICE),
        constraint_types=c_types.to(DEVICE),
        n_layers=PCAGAT_LAYERS,
        n_heads=PCAGAT_HEADS,
        dropout=PCAGAT_DROPOUT,
        use_constraint=True,
        use_kg=True,
        use_multi_head=True,
        use_gate=USE_CONSTRAINT_GATE,
        use_bi_interaction=USE_BI_INTERACTION,
        constraint_margin=CONSTRAINT_CONTRASTIVE_MARGIN,
        device=DEVICE
    ).to(DEVICE)

    if pretrain_user_emb is not None:
        model.load_pretrain(pretrain_user_emb, pretrain_item_emb,
                            dataset.item2entity)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
    print(f"\n   Model Statistics:")
    print(f"   |- Total parameters: {n_params:,}")
    print(f"   |- Trainable parameters: {n_trainable:,}")
    print(f"   |- GAT Layers: {PCAGAT_LAYERS}")
    print(f"   |- Attention Heads: {PCAGAT_HEADS}")
    print(f"   |- Embedding Dim: {EMBED_DIM}")
    print(f"   |- Bi-Interaction: {USE_BI_INTERACTION}")
    print(f"   |- Constraint Gate: {USE_CONSTRAINT_GATE}")
    print(f"   |- Contrastive Loss: {USE_CONSTRAINT_CONTRASTIVE}")
    print(f"   |- Alignment Loss: {USE_CONSTRAINT_ALIGNMENT}")
    print(f"   |- Hard Neg Sampling: {USE_HARD_NEG_SAMPLING}")
    print(f"   |- ★ Item CL Weight: {ITEM_CL_WEIGHT}")
    print(f"   |- ★ Item CL Neg Samples: {ITEM_CL_NEG_SAMPLES}")
    print(f"   |- ★ Item CL Start Epoch: {ITEM_CL_START_EPOCH}")

    print(f"\n{'='*70}")
    print("   Phase 2: Training PCA-GAT (V6.0)")
    print(f"{'='*70}")

    train_start = time.time()
    model, history = train_pcagat(
        model, dataset,
        contrastive_weight=(CONSTRAINT_CONTRASTIVE_WEIGHT
                            if USE_CONSTRAINT_CONTRASTIVE else 0),
        alignment_weight=(CONSTRAINT_ALIGNMENT_WEIGHT
                          if USE_CONSTRAINT_ALIGNMENT else 0),
        use_hard_neg=USE_HARD_NEG_SAMPLING,
        hard_neg_ratio=HARD_NEG_RATIO,
        item_cl_weight=ITEM_CL_WEIGHT,
        item_cl_neg_samples=ITEM_CL_NEG_SAMPLES,
        item_cl_start_epoch=ITEM_CL_START_EPOCH,
    )
    train_time = time.time() - train_start
    print(f"\n   Training time: {train_time:.1f}s ({train_time/60:.1f}min)")

    print(f"\n{'=' * 70}")
    print("   Phase 3: Full Evaluation")
    print(f"{'=' * 70}")

    model.eval()
    results = evaluate_model(model, dataset, TOP_K_LIST)
    print_results(results, 'PCA-GAT (V6.0 Bi-Interaction)')

    print("\n   Computing Constraint Satisfaction Rate...")
    for k in TOP_K_LIST:
        csr_results = compute_constraint_satisfaction(model, dataset, top_k=k)
        print(f"   CSR@{k}: {csr_results['csr']:.4f} "
              f"(violations: {csr_results['total_violations']})")

    csr_all = compute_constraint_satisfaction(
        model, dataset, top_k=max(TOP_K_LIST))

    save_results(results, 'PCAGAT_V6', extra_info={
        'training_time': train_time,
        'n_params': n_params,
        'csr': csr_all,
        'config': {
            'n_layers': PCAGAT_LAYERS,
            'n_heads': PCAGAT_HEADS,
            'embed_dim': EMBED_DIM,
            'dropout': PCAGAT_DROPOUT,
            'kg_weight': PCAGAT_KG_WEIGHT,
            'use_gate': USE_CONSTRAINT_GATE,
            'use_bi_interaction': USE_BI_INTERACTION,
            'item_cl_weight': ITEM_CL_WEIGHT,
        }
    })

    print(f"\n{'='*70}")
    print("   Phase 4: Constraint Weight Analysis")
    print(f"{'='*70}")

    constraint_weights = model.get_constraint_weights()
    if constraint_weights:
        print("\n   Learned Constraint Weights:")
        for layer_key, weights in constraint_weights.items():
            print(f"\n   {layer_key}:")
            lg = weights.get('lambda_global', 'N/A')
            if isinstance(lg, float):
                print(f"     Global lambda: {lg:.4f}")
            if 'lambda_c' in weights:
                lc = weights['lambda_c']
                if hasattr(lc, '__len__'):
                    for i, ctype in enumerate(CONSTRAINT_TYPES):
                        val = lc[i] if i < len(lc) else 0
                        print(f"     λ_{ctype}: {val:.4f}")
            if weights.get('has_gate'):
                print(f"     Gate: Active ✓")

        analysis_path = os.path.join(
            RESULT_DIR, 'pcagat_constraint_analysis.json')
        json_weights = {}
        for k, v in constraint_weights.items():
            jw = {}
            if 'lambda_c' in v:
                jw['lambda_c'] = (v['lambda_c'].tolist()
                                  if hasattr(v['lambda_c'], 'tolist')
                                  else list(v['lambda_c']))
            if 'lambda_global' in v:
                jw['lambda_global'] = float(v['lambda_global'])
            jw['has_gate'] = v.get('has_gate', False)
            json_weights[k] = jw
        with open(analysis_path, 'w') as f:
            json.dump(json_weights, f, indent=2)
        print(f"\n   Saved to {analysis_path}")

    print(f"\n{'=' * 70}")
    print("   Phase 5: Explainability Analysis")
    print(f"{'=' * 70}")

    if USE_AMAZON:
        print("   Skipping: Amazon dataset has no semantic entity names.")
    else:
        try:
            from explainer import PCAGATExplainer
            explainer = PCAGATExplainer(model, dataset)

            test_pairs = []
            for part_id, plan_ids in dataset.test_dict.items():
                for plan_id in plan_ids[:2]:
                    test_pairs.append((part_id, plan_id))
                if len(test_pairs) >= 20:
                    break

            if test_pairs:
                print("\n   === Case Studies ===")
                for part_id, plan_id in test_pairs[:3]:
                    exp = explainer.explain_recommendation(part_id, plan_id)
                    explainer.print_explanation(exp)

                print("\n   === Overall Metrics ===")
                exp_metrics = explainer.compute_explainability_metrics(test_pairs)
                explainer.print_metrics(exp_metrics)

                exp_path = os.path.join(RESULT_DIR, 'pcagat_explanations.json')
                all_explanations = []
                for part_id, plan_id in test_pairs:
                    exp = explainer.explain_recommendation(part_id, plan_id)
                    serializable_exp = {
                        'part': exp['part'],
                        'part_id': int(exp['part_id']),
                        'plan': exp['plan'],
                        'plan_id': int(exp['plan_id']),
                        'score': float(exp['score']),
                        'paths': []
                    }
                    for p in exp['paths']:
                        serializable_exp['paths'].append({
                            'node_names': p['node_names'],
                            'relation_names': p['relation_names'],
                            'attention_score': float(p['attention_score']),
                            'hops': int(p['hops']),
                            'path_string': p['path_string'],
                            'path_type': p['path_type']
                        })
                    all_explanations.append(serializable_exp)

                output = {
                    'explanations': all_explanations,
                    'metrics': {
                        k: (float(v) if isinstance(v, (int, float, np.floating))
                            else v)
                        for k, v in exp_metrics.items()
                    }
                }
                with open(exp_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"\n   Saved to {exp_path}")
        except Exception as e:
            print(f"   Explainability analysis failed: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"   PCA-GAT V6.0 TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"\n   Generated files:")
    print(f"   |- {PCAGAT_CHECKPOINT}")
    print(f"   |- {RESULT_DIR}/pcagat_v6_results.json")
    print(f"   |- {RESULT_DIR}/pcagat_constraint_analysis.json")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()