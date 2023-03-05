import torch

def get_nll_loss(pred, target):
  prob_of_correct_class = pred[torch.arange(len(target)), target]
  return -torch.log(prob_of_correct_class).mean()