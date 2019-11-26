import torch
from torch import nn

class RankingLoss(nn.Module):

    def __init__(self, margin=1, tie_margin=1):
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.tie_margin = tie_margin

    def forward(self,x_left,x_right,y):
        diff = x_left - x_right
        bin_y = torch.abs(y)
        tie_y =  1 - bin_y
        tie_loss = torch.clamp((torch.abs(diff) - self.tie_margin)*tie_y , min=0)
        rank_loss = torch.clamp(-y*(diff) + bin_y*self.margin, min=0)
        return torch.mean(rank_loss + tie_loss)




if __name__ == '__main__':
    y = torch.Tensor([-1,0,1])
    x_left = torch.Tensor([-0.000000001,-3,-0.000000001])
    x_right = torch.Tensor([-0.000000002,-5,-0.000000002])
    x_left.requires_grad_()
    x_right.requires_grad_()
    loss = RankingLoss(margin=5)
    result= loss(x_left,x_right,y)
    print(result)
    result.backward()