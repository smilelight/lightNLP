import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from ..config import DEVICE


def get_score(model, x, y, field_x, field_y, score_type='f1'):
    metrics_map = {
        'f1': f1_score,
        'p': precision_score,
        'r': recall_score,
        'acc': accuracy_score
    }
    # print('x', x)
    # print('y', y)
    metric_func = metrics_map[score_type] if score_type in metrics_map else metrics_map['f1']
    vec_x = torch.tensor([field_x.stoi[i] for i in x])
    predict_y = model(vec_x.view(-1, 1).to(DEVICE))[0]
    true_y = [field_y.stoi[i] for i in y]
    assert len(true_y) == len(predict_y)
    # print('predict_y', predict_y)
    # print('true_y', true_y)
    return metric_func(predict_y, true_y, average='micro')
