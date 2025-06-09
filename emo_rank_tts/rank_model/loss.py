import torch
import torch.nn as nn
import torch.nn.functional as F


########################################################################
# Ranking Loss Function
########################################################################
class RankLoss(nn.Module):

    def __init__(self, alpha, beta):
        super(RankLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions, targets):
        """Compute the value of the loss function

        Arguments
        ---------
        predictions: tuple
            model predictions
        targets: tuple
            ground truth targets
        
        Returns
        -------
        loss: torch.Tensor
            computed loss value
        L_mixup: torch.Tensor
            mixup classification loss
        L_rank: torch.Tensor
            ranking loss
        """
        
        y_emo, y_neu = targets
        lam_i, lam_j, _, _, hi, hj, ri, rj = predictions

        # mixup loss
        Li = lam_i.squeeze() * F.cross_entropy(hi, y_emo) \
            + (1 - lam_i.squeeze()) * F.cross_entropy(hi, y_neu)
        Lj = lam_j.squeeze() * F.cross_entropy(hj, y_emo) \
            + (1 - lam_j.squeeze()) * F.cross_entropy(hj, y_neu)
        L_mixup = (Li + Lj).mean()

        # ranking loss
        ri = ri.squeeze(-1)  # (B,)
        rj = rj.squeeze(-1)
        pij = torch.sigmoid(ri - rj)  # (B,)
        lam_diff = (lam_i.squeeze() - lam_j.squeeze() + 1) / 2
        L_rank = - (lam_diff * torch.log(pij + 1e-8) + (1 - lam_diff) * torch.log(1 - pij + 1e-8)).mean()

        # total loss
        loss = self.alpha * L_mixup + self.beta * L_rank
        return loss, L_mixup, L_rank
