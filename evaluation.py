import torch


def eval_every_timestep(true_batch, pred_batch):
    # true_batch:(S),pred_batch:(S,L)
    _, top_k_preds = pred_batch.topk(pred_batch.size(1), largest=True)
    l = len(pred_batch)
    true_batch = true_batch.view(-1, 1)  # (S,1)
    acc1 = torch.sum(top_k_preds[:, :1] == true_batch).item()
    acc5 = torch.sum(top_k_preds[:, :5] == true_batch).item()
    acc10 = torch.sum(top_k_preds[:, :10] == true_batch).item()
    acc20 = torch.sum(top_k_preds[:, :20] == true_batch).item()
    matches = (top_k_preds == true_batch).nonzero()[:, 1] + 1
    reciprocal_ranks = torch.reciprocal(matches.to(torch.float))
    mrr = torch.sum(reciprocal_ranks)
    return acc1, acc5, acc10, acc20, mrr, l
