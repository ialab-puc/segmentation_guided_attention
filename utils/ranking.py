import numpy as np
from sklearn.metrics import label_ranking_average_precision_score as rank_score

def compute_ranking_loss(x_left,x_right,label,loss):
    output_left = x_left.view(x_left.size()[0])
    output_right = x_right.view(x_right.size()[0])
    return loss(output_left, output_right, label)

def compute_ranking_accuracy(x_left, x_right, label):
    rank_pairs = np.array(list(zip(x_left,x_right)))
    label_matrix = label.clone().cpu().detach().numpy()
    dup = np.zeros(label_matrix.shape)
    label_matrix[label_matrix==-1] = 0
    dup[label_matrix==0] = 1
    label_matrix = np.hstack((np.array([label_matrix]).T,np.array([dup]).T))
    return  (rank_score(label_matrix,rank_pairs) - 0.5)/0.5