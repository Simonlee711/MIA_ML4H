import numpy as np
from sklearn.metrics import roc_auc_score


def empirical_auc(member_score, non_member_score):
    auc = 0
    for s_i in member_score:
        for s_j in non_member_score:
            if s_i > s_j:
                auc += 1
            elif s_i == s_j:
                auc += 0.5
    
    auc /= (len(member_score) * len(non_member_score))
    return auc


def empirical_advantage(member_score, non_member_score, t):
    adv_member= np.sum(member_score >= t) / len(member_score)
    adv_non_member = np.sum(non_member_score >= t) / len(non_member_score)

    advantage = adv_member - adv_non_member

    return advantage
   


def empirical_auc_sklearn(member_score, non_member_score):
    scores = np.concatenate([member_score, non_member_score])
    labels = np.concatenate([np.ones_like(member_score), np.zeros_like(non_member_score)])
    return roc_auc_score(labels, scores)

