import torch
from model_utils import mc_forward
from tqdm.auto import tqdm

def entropy(probs):
    return -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1)

def bald(probs):
    entropy_mean = entropy(probs.mean(dim=0)) 
    expected_entropy = entropy(probs).mean(dim=0)
    return entropy_mean - expected_entropy


def variation_ratio(probs):
    mean_probs = probs.mean(dim=0)
    return 1.0 - mean_probs.max(dim=-1).values

def mean_std(probs):
    mean = probs.mean(dim=0) 
    squared_mean = (probs**2).mean(dim=0) 
    std = (squared_mean - mean**2).clamp_min(0).sqrt() 
    return std.mean(dim=-1) 


def score_acquisition(logits, acquisition):
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    probs = logits.softmax(dim=-1)
    acq_map = {
        "bald": lambda p: bald(p),
        "variation": lambda p: variation_ratio(p),
        "entropy": lambda p: entropy(p.mean(dim=0)),
        "mean_std": lambda p: mean_std(p),
    }
    fn = acq_map.get(acquisition.lower())
    if fn is None:
        raise ValueError(f"Unknown acquisition function {acquisition}")
    return fn(probs)

def score_pool(model, loader, device, T, acquisition, use_dropout=True, use_tqdm=False, return_embeddings=False):
    model.eval()
    if use_dropout and T > 1:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    scores = []
    embs = [] if return_embeddings else None
    with torch.inference_mode():
        iter_loader = tqdm(loader, desc="pool batches", leave=False, disable=not use_tqdm)
        for x, _ in iter_loader:
            x = x.to(device, non_blocking=True)
            logits, _ = mc_forward(model, x, T, use_dropout)  
            scores.append(score_acquisition(logits, acquisition)) 
            if return_embeddings:
                with torch.no_grad():
                    emb = model.forward_features(x)
                embs.append(emb)
    scores_cat = torch.cat(scores, dim=0)
    if return_embeddings:
        embs_cat = torch.cat(embs, dim=0)
        return scores_cat, embs_cat
    return scores_cat


def select_top_k(scores, pool_idx, k, rng):
    if not pool_idx:
        return []
    pool_idx = list(pool_idx)
    k = min(k, len(pool_idx))
    perm = torch.as_tensor(rng.permutation(len(pool_idx)), device=scores.device)
    top = perm[scores[perm].topk(k).indices].tolist()
    return [pool_idx[i] for i in top]


def select_batch_diverse(scores, embs, pool_idx, k, diversity_lambda, rng):
    if not pool_idx:
        return []
    pool_idx = list(pool_idx)
    k = min(k, len(pool_idx))

    device = scores.device
    scores = scores.clone()
    embs = embs.clone()

    embs = embs / (embs.norm(dim=1, keepdim=True) + 1e-12)

    selected = []
    selected_mask = torch.zeros(len(pool_idx), device=device, dtype=torch.bool)

    for _ in range(k):
        if selected:
            sel = embs[selected]
            sims = embs @ sel.T
            penalty = sims.max(dim=1).values
        else:
            penalty = torch.zeros_like(scores)

        combined = scores - diversity_lambda * penalty
        combined[selected_mask] = -float("inf")

        idx = combined.argmax().item()
        selected.append(idx)
        selected_mask[idx] = True

    return [pool_idx[i] for i in selected]
