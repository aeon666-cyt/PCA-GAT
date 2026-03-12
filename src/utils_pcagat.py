import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import json
import random
import os
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

from config_pcagat import *


def compute_metrics(ranked_list, ground_truth, k):
    hits = 0
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            hits += 1
            dcg += 1.0 / np.log2(i + 2)
    recall = hits / len(ground_truth) if ground_truth else 0
    precision = hits / k
    hr = 1.0 if hits > 0 else 0.0
    idcg = sum([1.0 / np.log2(i + 2)
                for i in range(min(len(ground_truth), k))])
    ndcg = dcg / idcg if idcg > 0 else 0
    return recall, precision, hr, ndcg


def compute_mrr(ranked_list, ground_truth, k):
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_model(model, dataset, top_k_list=TOP_K_LIST, max_users=None):
    model.eval()
    results = {k: {'Recall': [], 'HR': [], 'NDCG': [], 'Precision': [], 'MRR': []}
               for k in top_k_list}

    users = list(dataset.test_dict.keys())
    if max_users is not None and max_users < len(users):
        users = random.sample(users, max_users)

    with torch.no_grad():
        if hasattr(model, '_propagate'):
            final_emb = model._propagate(return_attention=False)
            user_emb = final_emb[:model.n_users]
            entity_emb = final_emb[model.n_users:]
            item_emb = entity_emb[model._item_indices]
        elif model.__class__.__name__ == 'KGAT':
            user_emb, entity_emb = model()
            item_emb = model.get_item_embeddings(entity_emb)
        elif model.__class__.__name__ == 'KGIN':
            if hasattr(model, 'get_full_embeddings'):
                user_emb, item_emb = model.get_full_embeddings()
            else:
                kg_entity_emb = model._propagate_kg()
                item_kg_emb = kg_entity_emb[model.item2entity_map]
                user_emb, item_cf = model._propagate_ui(
                    model.user_emb.weight, item_kg_emb)
                item_emb = item_cf + item_kg_emb
        elif model.__class__.__name__ == 'CKE':
            user_emb = model.user_emb.weight
            item_emb = model.get_item_embeddings()
        elif hasattr(model, 'computer'):
            user_emb, item_emb = model.computer()
        elif hasattr(model, 'user_emb') and hasattr(model, 'item_emb'):
            user_emb = model.user_emb.weight
            item_emb = model.item_emb.weight
        else:
            raise ValueError(f"不支持的模型类型: {model.__class__.__name__}")

        test_user_emb = user_emb[users]
        all_scores = torch.matmul(test_user_emb, item_emb.t()).cpu().numpy()

        for idx, u in enumerate(users):
            scores = all_scores[idx].copy()
            trained_items = list(dataset.train_set.get(u, []))
            if trained_items:
                scores[trained_items] = -1e9
            ranked_items = np.argsort(scores)[::-1]
            ground_truth = set(dataset.test_dict[u])
            if not ground_truth:
                continue
            for k in top_k_list:
                recall, precision, hr, ndcg = compute_metrics(
                    ranked_items[:k], ground_truth, k)
                mrr = compute_mrr(ranked_items, ground_truth, k)
                results[k]['Recall'].append(recall)
                results[k]['Precision'].append(precision)
                results[k]['HR'].append(hr)
                results[k]['NDCG'].append(ndcg)
                results[k]['MRR'].append(mrr)

    avg_results = {}
    for k in top_k_list:
        avg_results[k] = {
            'Recall':    np.mean(results[k]['Recall'])    if results[k]['Recall']    else 0,
            'Precision': np.mean(results[k]['Precision']) if results[k]['Precision'] else 0,
            'HR':        np.mean(results[k]['HR'])        if results[k]['HR']        else 0,
            'NDCG':      np.mean(results[k]['NDCG'])      if results[k]['NDCG']      else 0,
            'MRR':       np.mean(results[k]['MRR'])       if results[k]['MRR']       else 0,
        }
    return avg_results


def compute_constraint_satisfaction(model, dataset, top_k=5):
    model.eval()
    entity_constraints = dataset.constraint_rules.get(
        'entity_type_constraints', {})

    neg_pairs = set()
    for ctype, pairs in entity_constraints.items():
        for pair_key, score in pairs.items():
            if score < 0:
                pk = pair_key.split(',')
                if len(pk) == 2:
                    neg_pairs.add((int(pk[0]), int(pk[1])))

    if not neg_pairs:
        return {'csr': 1.0, 'violation_rate': 0.0,
                'total_violations': 0, 'total_pairs': 0}

    item_neighbor_ents = {}
    for item_id in range(dataset.n_items):
        ent_id = dataset.item2entity.get(item_id)
        if ent_id is not None:
            neighbors = {ent_id}
            for r, t in dataset.kg_dict.get(ent_id, []):
                if t < dataset.n_entities:
                    neighbors.add(t)
            item_neighbor_ents[item_id] = neighbors

    total_recs = 0
    satisfied_recs = 0
    total_violations = 0
    users = list(dataset.test_dict.keys())

    with torch.no_grad():
        if hasattr(model, '_propagate'):
            final_emb = model._propagate(return_attention=False)
            user_emb = final_emb[:model.n_users]
            entity_emb = final_emb[model.n_users:]
            item_emb = entity_emb[model._item_indices]
        elif model.__class__.__name__ == 'KGAT':
            user_emb, entity_emb = model()
            item_emb = model.get_item_embeddings(entity_emb)
        elif hasattr(model, 'computer'):
            user_emb, item_emb = model.computer()
        elif hasattr(model, 'user_emb') and hasattr(model, 'item_emb'):
            user_emb = model.user_emb.weight
            item_emb = model.item_emb.weight
        else:
            return {'csr': 1.0, 'violation_rate': 0.0,
                    'total_violations': 0, 'total_pairs': 0}

        for u in users:
            user_entities = set()
            for item in dataset.train_set.get(u, []):
                ent_id = dataset.item2entity.get(item)
                if ent_id is not None:
                    user_entities.add(ent_id)
                    for r, t in dataset.kg_dict.get(ent_id, []):
                        if t < dataset.n_entities:
                            user_entities.add(t)
            if not user_entities:
                continue

            u_emb = user_emb[u:u+1]
            scores = torch.matmul(u_emb, item_emb.t()).cpu().numpy()[0]
            for i in dataset.train_set.get(u, []):
                scores[i] = -1e9
            top_items = np.argsort(scores)[::-1][:top_k]

            for item_id in top_items:
                item_ents = item_neighbor_ents.get(int(item_id), set())
                if not item_ents:
                    continue
                has_violation = False
                n_violations = 0
                for ue in user_entities:
                    for ie in item_ents:
                        if (ue, ie) in neg_pairs or (ie, ue) in neg_pairs:
                            has_violation = True
                            n_violations += 1
                total_recs += 1
                if not has_violation:
                    satisfied_recs += 1
                total_violations += n_violations

    csr = satisfied_recs / total_recs if total_recs > 0 else 1.0
    violation_rate = total_violations / total_recs if total_recs > 0 else 0
    return {
        'csr': csr,
        'violation_rate': violation_rate,
        'total_recommendations': total_recs,
        'satisfied_recommendations': satisfied_recs,
        'total_violations': total_violations,
    }


def print_results(results, model_name='Model'):
    print(f"\n   {'='*72}")
    print(f"   📊 {model_name} Results")
    print(f"   {'='*72}")
    print(f"   {'K':>4} | {'Recall':>10} | {'Precision':>10} | "
          f"{'HR':>10} | {'NDCG':>10} | {'MRR':>10}")
    print(f"   {'-'*72}")
    for k in sorted(results.keys()):
        r = results[k]
        print(f"   {k:>4} | {r['Recall']:>10.4f} | "
              f"{r['Precision']:>10.4f} | "
              f"{r['HR']:>10.4f} | {r['NDCG']:>10.4f} | "
              f"{r.get('MRR', 0):>10.4f}")
    print(f"   {'='*72}")


def save_results(results, model_name, extra_info=None):
    output = {
        'model': model_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': {str(k): v for k, v in results.items()},
    }
    if extra_info:
        output.update(extra_info)
    filepath = os.path.join(RESULT_DIR, f'{model_name.lower()}_results.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Results saved to {filepath}")


def pretrain_bprmf(dataset, dim=EMBED_DIM, epochs=PRETRAIN_EPOCHS,
                   lr=PRETRAIN_LR, patience=PRETRAIN_PATIENCE):
    from model_pcagat import BPRMF_Pretrain

    print("\n" + "=" * 70)
    print("   Phase 0: BPR-MF Pre-training")
    print("=" * 70)

    model = BPRMF_Pretrain(dataset.n_users, dataset.n_items, dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    n_batches = max(1, len(dataset.train_data) // BATCH_SIZE)

    best_recall = 0.0
    patience_counter = 0
    best_state = None

    for epoch in tqdm(range(epochs), desc="Pretrain", ncols=100):
        model.train()
        total_loss = 0

        for _ in range(n_batches):
            users, pos_items, neg_items = _sample_bpr_batch(dataset, batch_size=BATCH_SIZE)

            optimizer.zero_grad()
            pos, neg = model(users, pos_items, neg_items)
            loss = -torch.mean(F.logsigmoid(pos - neg))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                user_embs = model.user_emb.weight
                item_embs = model.item_emb.weight
                recalls = []
                test_users = list(dataset.test_dict.keys())
                if len(test_users) > 5000:
                    test_users = random.sample(test_users, 5000)
                for u in test_users:
                    scores = torch.matmul(
                        user_embs[u:u+1], item_embs.t()).cpu().numpy()[0]
                    for i in dataset.train_set.get(u, []):
                        scores[i] = -1e9
                    top5 = np.argsort(scores)[::-1][:5]
                    gt = set(dataset.test_dict[u])
                    if gt:
                        recalls.append(len(set(top5) & gt) / len(gt))
                current_recall = np.mean(recalls) if recalls else 0

            print(f"   Epoch {epoch+1:4d}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Recall@5={current_recall:.4f}", end='')
            if current_recall > best_recall + 1e-4:
                best_recall = current_recall
                patience_counter = 0
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                print(" ✅ [BEST]")
            else:
                patience_counter += 1
                print(f" (patience: {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n   ⏹ Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    print(f"\n   ✓ Pre-training complete. Best Recall@5: {best_recall:.4f}")
    torch.save({
        'user_emb': model.user_emb.weight.detach().cpu(),
        'item_emb': model.item_emb.weight.detach().cpu(),
        'best_recall': float(best_recall),
    }, PRETRAIN_CHECKPOINT)
    print(f"   ✓ Saved to {PRETRAIN_CHECKPOINT}")
    return model.user_emb.weight.detach(), model.item_emb.weight.detach()


def _sample_bpr_batch(dataset, batch_size=65536):
    n_train = len(dataset.train_data)
    idx = np.random.randint(0, n_train, size=batch_size)
    pairs = dataset.train_data[idx]
    users = pairs[:, 0]
    pos_items = pairs[:, 1]
    neg_items = np.random.randint(0, dataset.n_items, size=batch_size)
    for i in range(batch_size):
        u = users[i]
        while neg_items[i] in dataset.train_set.get(u, set()):
            neg_items[i] = np.random.randint(0, dataset.n_items)
    return (torch.LongTensor(users).to(DEVICE),
            torch.LongTensor(pos_items).to(DEVICE),
            torch.LongTensor(neg_items).to(DEVICE))


def train_pcagat(model, dataset, epochs=PCAGAT_EPOCHS, lr=PCAGAT_LR,
                 kg_weight=PCAGAT_KG_WEIGHT, l2_weight=PCAGAT_L2_WEIGHT,
                 contrastive_weight=CONSTRAINT_CONTRASTIVE_WEIGHT,
                 alignment_weight=CONSTRAINT_ALIGNMENT_WEIGHT,
                 use_hard_neg=USE_HARD_NEG_SAMPLING,
                 hard_neg_ratio=HARD_NEG_RATIO,
                 patience=PCAGAT_PATIENCE, min_epochs=PCAGAT_MIN_EPOCHS,
                 eval_interval=PCAGAT_EVAL_INTERVAL,
                 checkpoint_path=PCAGAT_CHECKPOINT, verbose=True,
                 item_cl_weight=0.0,
                 item_cl_neg_samples=64,
                 item_cl_start_epoch=10):

    es_metric = EARLY_STOP_METRIC
    if es_metric == 'R@20':
        es_k = 20
    elif es_metric == 'R@10':
        es_k = 10
    elif es_metric == 'R@5':
        es_k = 5
    else:
        es_k = 1

    n_batches_original = max(1, len(dataset.train_data) // BATCH_SIZE)
    total_steps = epochs * n_batches_original
    actual_eval_interval = eval_interval * n_batches_original
    actual_min_steps = min_epochs * n_batches_original

    if verbose:
        print(f"\n   原始: {epochs} epochs × {n_batches_original} batches = {total_steps} steps")
        print(f"   V4.0: {total_steps} steps, 每step=1次传播+1batch")
        print(f"   评估间隔: 每 {actual_eval_interval} steps (≈{eval_interval} epochs)")
        print(f"   ★ Early stop metric: {es_metric} (k={es_k})")

    embedding_params = []
    gnn_params = []
    constraint_params = []

    for name, param in model.named_parameters():
        if ('lambda_c' in name or 'lambda_global' in name
                or 'constraint_gate' in name or 'cl_temperature' in name):
            constraint_params.append(param)
        elif 'emb' in name:
            embedding_params.append(param)
        else:
            gnn_params.append(param)

    optimizer = optim.Adam([
        {'params': embedding_params, 'lr': lr * 0.1},
        {'params': gnn_params, 'lr': lr, 'weight_decay': 1e-4},
        {'params': constraint_params, 'lr': lr * PCAGAT_CONSTRAINT_LR_MULT},
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
    )

    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    kg_loader = dataset.get_kg_loader() if model.use_kg else None
    kg_iter = iter(kg_loader) if kg_loader else None

    use_contrastive = (contrastive_weight > 0 and model.use_constraint
                       and model.constraint_indices.size(1) > 0)
    use_alignment = (alignment_weight > 0 and model.use_constraint
                     and model.constraint_indices.size(1) > 0
                     and USE_CONSTRAINT_ALIGNMENT)
    use_item_cl = (item_cl_weight > 0
                   and hasattr(model, 'compute_item_cl_loss'))

    best_metric = 0.0
    patience_counter = 0
    best_model_state = None
    history = {
        'loss': [], 'bpr_loss': [], 'kg_loss': [], 'cl_loss': [],
        'recall_1': [], 'recall_5': [], 'recall_20': [], 'mrr_1': []
    }

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"   Training PCA-GAT (V4.0) - Early stop: {es_metric}")
        print(f"{'=' * 70}")

    bpr_sample_size = min(65536, len(dataset.train_data))

    pbar = tqdm(range(total_steps), desc="Training", ncols=100)

    for step in pbar:
        model.train()
        optimizer.zero_grad()

        orig_epoch = step // n_batches_original
        epoch_use_cl = use_item_cl and (orig_epoch >= item_cl_start_epoch)

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            final_emb = model._propagate(return_attention=False)
            user_emb_all = final_emb[:model.n_users]
            entity_emb_all = final_emb[model.n_users:]
            item_emb_all = entity_emb_all[model._item_indices]

            users, pos_items, neg_items = _sample_bpr_batch(
                dataset, batch_size=bpr_sample_size)

            u_e = user_emb_all[users]
            pos_e = item_emb_all[pos_items]
            neg_e = item_emb_all[neg_items]
            bpr_loss = -F.logsigmoid(
                (u_e * pos_e).sum(1) - (u_e * neg_e).sum(1)).mean()

            kg_loss = torch.tensor(0.0, device=DEVICE)
            if kg_loader is not None and kg_weight > 0:
                try:
                    kg_batch = next(kg_iter)
                except StopIteration:
                    kg_iter = iter(kg_loader)
                    kg_batch = next(kg_iter)
                kg_batch = kg_batch.to(DEVICE)
                kg_loss = model.compute_kg_loss(
                    kg_batch[:, 0], kg_batch[:, 1],
                    kg_batch[:, 2], kg_batch[:, 3])

            l2_loss = model.compute_l2_loss(users, pos_items, neg_items)

            con_loss = (model.compute_constraint_contrastive_loss()
                        if use_contrastive
                        else torch.tensor(0.0, device=DEVICE))
            align_loss = (model.compute_constraint_alignment_loss(
                sample_size=CONSTRAINT_ALIGNMENT_SAMPLE)
                          if use_alignment
                          else torch.tensor(0.0, device=DEVICE))

            cl_loss = torch.tensor(0.0, device=DEVICE)
            if epoch_use_cl:
                cl_loss = model.compute_item_cl_loss(
                    users, pos_items,
                    train_set=dataset.train_set,
                    n_neg_samples=item_cl_neg_samples)

            loss = (bpr_loss
                    + kg_weight * kg_loss
                    + l2_weight * l2_loss
                    + contrastive_weight * con_loss
                    + alignment_weight * align_loss
                    + item_cl_weight * cl_loss)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'bpr': f'{bpr_loss.item():.3f}',
            'kg': f'{kg_loss.item():.3f}',
        })

        if (step + 1 >= actual_min_steps
                and (step + 1) % actual_eval_interval == 0):
            model.eval()
            with torch.no_grad():
                metrics = evaluate_model(
                    model, dataset, top_k_list=[1, 5, 10, 20])
                r1  = metrics[1]['Recall']
                r5  = metrics[5]['Recall']
                r10 = metrics[10]['Recall']
                r20 = metrics[20]['Recall']
                n20 = metrics[20]['NDCG']
                m1  = metrics[1]['MRR']

            current_metric = metrics[es_k]['Recall']

            if verbose:
                ep = (step + 1) // n_batches_original
                log_str = (f"\n   Eval (≈epoch {ep}): "
                           f"R@1={r1:.4f} R@5={r5:.4f} "
                           f"R@10={r10:.4f} R@20={r20:.4f} | "
                           f"N@20={n20:.4f}")
                cw = model.get_constraint_weights()
                for lv in cw.values():
                    if 'lambda_global' in lv:
                        lg = lv['lambda_global']
                        lc = lv.get('lambda_c', [])
                        lc_str = ','.join(f'{v:.2f}' for v in lc)
                        log_str += f" | λg={lg:.3f} λc=[{lc_str}]"
                        break
                print(log_str, end='')

            if current_metric > best_metric + 1e-4:
                best_metric = current_metric
                patience_counter = 0
                best_model_state = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}
                if verbose:
                    print(f" ✅ [BEST {es_metric}={current_metric:.4f}]")
            else:
                patience_counter += 1
                if verbose:
                    print(f" (patience: {patience_counter}/{patience})")

            scheduler.step(current_metric)

            if patience_counter >= patience:
                if verbose:
                    print(f"\n   ⏹ Early stopping at step {step+1}")
                break

    pbar.close()

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, history


def train_rec_model(model, loader, epochs, lr, dataset=None,
                    model_name='Model', patience=15,
                    checkpoint_path=None, verbose=True):
    opt = optim.Adam(model.parameters(), lr=lr)
    best_recall = 0.0
    patience_counter = 0
    best_model_state = None
    history = {'loss': [], 'recall': []}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"   Training {model_name}")
        print(f"{'=' * 70}")
        print(f"   📝 Loader has {len(loader)} batches")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pos, neg = model(batch[:, 0], batch[:, 1], batch[:, 2])
            loss = -torch.mean(F.logsigmoid(pos - neg))
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
        history['loss'].append(avg_loss)

        if (epoch + 1) % 5 == 0 and dataset is not None:
            model.eval()
            with torch.no_grad():
                metrics = evaluate_model(model, dataset, top_k_list=[10])
                current_recall = metrics[10]['Recall']
            history['recall'].append(current_recall)
            if verbose:
                print(f"   Epoch {epoch+1:4d}/{epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Recall@10={current_recall:.4f}", end='')
            if current_recall > best_recall + 1e-4:
                best_recall = current_recall
                patience_counter = 0
                best_model_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()}
                if verbose:
                    print(" ✅ [BEST]")
            else:
                patience_counter += 1
                if verbose:
                    print(f" (patience: {patience_counter}/{patience})")
            if patience_counter >= patience:
                if verbose:
                    print(f"\n   ⏹ Early stopping at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    if checkpoint_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_recall': best_recall,
            'history': history,
        }, checkpoint_path)
    return model, history