import torch
from torch import nn

class RankingLoss(nn.Module):

    def __init__(self, margin=1):
        super(RankingLoss, self).__init__()
        self.margin = 1

    def forward(self,y,x_left,x_right):
        diff = x_left - x_right
        bin_y = torch.abs(y)
        tie_y =  1 - bin_y
        tie_loss = torch.mean(torch.clamp(torch.abs(diff)*tie_y, min=0))
        rank_loss = torch.mean(torch.clamp(-y*(diff) + bin_y*self.margin, min=0))
        return tie_loss + rank_loss




if __name__ == '__main__':
    y = torch.Tensor([-1,0])
    x_left = torch.Tensor([2.5,-3])
    x_right = torch.Tensor([1,-2])
    x_left.requires_grad_()
    x_right.requires_grad_()
    loss = RankingLoss()
    result= loss(y,x_left,x_right)
    print(result)
    result.backward()