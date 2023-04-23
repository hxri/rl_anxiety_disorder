import torch
import torch.nn.functional as F
import numpy as np

def NormalizeData(data, min, max):
    return (data - min) / (max - min)


def motivational_relevance(obs):
    """Computes motivational relevance for a batch of observations.
    Motivational relevance is a function of the L1 distance to the goal.
    Some observation states do not contain the goal, so relevance is zero.
    """
    min = 0.0
    max = 1.0

    # print(obs.shape)
    batch_size, w, _ = obs.size()
    relevance = torch.zeros(batch_size)
    agent_pos = torch.nonzero(obs == 10)[:, 1:]
    goal_poss = torch.nonzero(obs == 8)
    for goal in goal_poss:
        idx, goal_pos = goal[0], goal[1:]
        dist = torch.norm(agent_pos[idx] - goal_pos.float(), 1)
        relevance[idx] = 1 - (dist - 1) / (2 * (w - 1))
    norm = NormalizeData(relevance, min, max)
    return norm

def novelty(logits):
    """Computes novelty according to the KL Divergence from perfect uncertainty.
    The higher the KL Divergence, the less novel the scenario,
    so we take novelty as the negative of the KL Divergence.
    """
    min = -4
    max = 0.0

    batch_size, num_actions = logits.size()
    P = torch.softmax(logits, dim=1)
    Q = torch.full(P.size(), 1 / num_actions)
    nov = -torch.sum(Q * torch.log(Q / P), dim=1)
    norm = NormalizeData(nov, min, max)
    return norm

def certainity(logits):
    """Computes the certainty of the action selection based on the entropy of the action probabilities.
    The higher the entropy, the less certain the action selection, so we take the negative of the entropy
    as the certainty score.
    """
    min = 0.0

    batch_size, num_actions = logits.size()
    max = -np.log(1/num_actions)
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    certainty = -entropy
    norm = NormalizeData(certainty, min, max)
    return norm

def coping_potential(logits):
    """Computes coping potential based on the variance of the action probabilities.
    The higher the variance, the greater the coping potential, so we use the variance
    as the coping potential score.
    """
    min = 0.0
    max = 1.0

    batch_size, num_actions = logits.size()
    probs = torch.softmax(logits, dim=1)
    mean_probs = torch.mean(probs, dim=1, keepdim=True)
    variance = torch.mean((probs - mean_probs) ** 2, dim=1)
    norm = NormalizeData(variance, min, max)
    return norm

def anticipation(logits):
    """Computes anticipation based on the predicted action values (logits).
    The higher the predicted future reward, the greater the anticipation, so we use the
    predicted action values to compute the anticipated reward for the current state.
    """
    min = 0.0
    max = 1.0

    batch_size, num_actions = logits.size()
    Q = logits.softmax(dim=1)
    V = Q.max(dim=1)[0]
    norm = NormalizeData(V, min, max)
    return norm

def goal_congruence(logits):
    """Computes goal congruence by comparing the predicted action distribution to a
    reference action distribution that represents a typical or common action distribution
    for the current state. The closer the predicted distribution is to the reference
    distribution, the greater the goal congruence.
    """
    min = 0.0
    max = 1.0

    batch_size, num_actions = logits.size()
    logits_norm = logits - torch.max(logits, dim=1, keepdim=True)[0]  # normalize logits
    P = logits_norm.softmax(dim=1)
    ref_dist = torch.full((batch_size, num_actions), 1 / num_actions)
    dist = torch.sum(torch.abs(P - ref_dist), dim=1)
    congruence = 1 - dist / (2 * (num_actions - 1))
    norm = NormalizeData(congruence, min, max)
    return norm



