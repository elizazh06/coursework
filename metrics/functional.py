import ast
from collections import defaultdict

import torch


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def macro_f1_multiclass(preds, labels, num_classes):
    eps = 1e-8
    scores = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        if tp == 0 and fp == 0 and fn == 0:
            continue
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        scores.append(2 * p * r / (p + r + eps))
    return float(sum(scores) / len(scores)) if scores else 0.0


def top3_hits(forward_pass, dataloader):
    hits = total = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs, labels, _ = forward_pass(batch)
            if labels.numel() == 0:
                continue
            k = min(3, outputs.size(1))
            topk = outputs.topk(k=k, dim=1).indices
            hits += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
    return hits, total


def by_question_type(preds, labels, question_types):
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for p, y, t in zip(preds.tolist(), labels.tolist(), question_types):
        try:
            parsed = ast.literal_eval(t) if isinstance(t, str) else t
            coarse = parsed[0] if len(parsed) > 0 else "Unknown"
            fine = parsed[1] if len(parsed) > 1 else "Unknown"
        except (ValueError, SyntaxError, TypeError):
            coarse, fine = "Unknown", "Unknown"
        key = f"{coarse}/{fine}"
        stats[key]["total"] += 1
        stats[coarse]["total"] += 1
        if p == y:
            stats[key]["correct"] += 1
            stats[coarse]["correct"] += 1
    return {
        k: (v["correct"] / v["total"]) if v["total"] else 0.0
        for k, v in sorted(stats.items())
    }
