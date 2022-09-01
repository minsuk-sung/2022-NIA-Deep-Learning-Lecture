import warnings
from sklearn.metrics import precision_score, recall_score

warnings.filterwarnings('ignore')

def accuracy(targets, preds, batch_size):
    correct = sum(targets == preds).cpu()
    acc = (correct/batch_size)
    return acc

def precision(targets, preds):
    targets = targets.detach().cpu().flatten()
    preds = preds.detach().cpu().flatten()
    return precision_score(targets, preds)

def recall(targets, preds):
    targets = targets.detach().cpu().flatten()
    preds = preds.detach().cpu().flatten()
    return recall_score(targets, preds)

