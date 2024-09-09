import torch

def softmax(x, dim=-1, temp=1):
    z = torch.exp((x - torch.max(x, dim=dim, keepdim=True).values) / temp)
    z_sum = z.sum(dim=dim, keepdim=True)
    return z / z_sum

def topk_accuracy(logits, targets, ks, mask=None):
    if isinstance(mask, torch.Tensor):
        real_logits = logits[mask]
        real_targets = targets[mask]
    else:
        real_logits = logits
        real_targets = targets

    count_real = len(real_targets)
    counts_correct = []
    for k in ks:
        topk_indices = torch.topk(real_logits, k).indices
        topk_correct = (topk_indices == real_targets.unsqueeze(1)).sum()
        counts_correct.append(topk_correct)

    return count_real, counts_correct