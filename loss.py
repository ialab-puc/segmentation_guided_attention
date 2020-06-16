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



class LogSumExpLoss(nn.Module):

    def forward(self,x_left,x_right,y):
        diff = x_left - x_right
        bin_y = torch.abs(y)
        tie_y =  1 - bin_y
        rank_loss = torch.log(1 + torch.exp(-y*diff)) * bin_y
        tie_loss = torch.log(torch.exp(diff) +  torch.exp(-diff)) * tie_y
        print(rank_loss, tie_loss)
        return torch.mean(rank_loss + tie_loss)

if __name__ == '__main__':
    ls = LogSumExpLoss()
    x_left = torch.Tensor([9])
    x_right = torch.Tensor([13])
    y = torch.Tensor([0])
    print(ls(x_left,x_right,y))
