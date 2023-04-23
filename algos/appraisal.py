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

def certainity(logits): # Entropy
    """
    Computes the coping potential of an RL agent given the logits predicted by the model.

    Args:
        logits: A tensor of shape (batch_size, num_actions) containing the logits for each action.

    Returns:
        A tensor of shape (batch_size,) containing the coping potential for each example in the batch.
    """
    min = 0.0
    max = 1.1


    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    norm = NormalizeData(entropy, min, max)
    return norm

def coping_potential(logits):
    """
    The coping potential measures the degree of control that the agent has over
    the environment, based on the difference between the expected and actual
    outcomes of its actions. A higher coping potential means that the agent has
    more control over the environment, while a lower coping potential means
    that the agent has less control.
    """
    min = -1.0
    max = 0.0

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)

    # Calculate expected reward
    expected_reward = torch.sum(probs * logits, dim=1)

    # Calculate actual reward
    actual_reward = torch.max(logits, dim=1).values

    # Calculate coping potential
    coping_pot = expected_reward - actual_reward
    norm = NormalizeData(coping_pot, min, max)
    return norm

def anticipation(logits):
    """
    The function takes in a tensor logits which represents the predicted logits
    for each action in a given state. We convert these logits to probabilities
    using the softmax function. Then, we calculate the entropy of the
    probabilities, which measures the amount of uncertainty or randomness in
    the probability distribution. The anticipation is defined as the inverse
    of the entropy, which gives a measure of the agent's level of confidence or
    expectation about the outcomes of the actions. Finally, we return the
    tensor of anticipations.
    """
    min = 0.0
    max = 100.0

    # Convert logits to probabilities using softmax function
    probs = torch.softmax(logits, dim=1)
    
    # Calculate the entropy of the probabilities
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    
    # Calculate the anticipation as the inverse of the entropy
    anticipation = 1.0 / entropy
    # Return the tensor of anticipations
    norm = NormalizeData(anticipation, min, max)
    return norm


def goal_congruence(logits):
    """
    To compute the goal congruence score for an RL agent in a grid world environment, goal congruence of an RL agent in
    a grid world environment using entropy, we can compute the entropy of the policy distribution over actions at each
    state in the grid world. The idea is that if the agent's policy is more focused on the actions that lead to the
    goal state, the entropy of the policy distribution should be lower.

    The function first converts the observation and logits to PyTorch tensors. It then computes the softmax of the
    logits to get the policy distribution over actions, and computes the entropy of the policy distribution using the
    formula -sum(p*log(p)), where p is the probability of each action. The function also computes the KL-divergence
    from the uniform distribution as a measure of how far the policy is from being uniformly distributed.

    Finally, the function computes the goal congruence score as the negative entropy of the policy distribution,
    adjusted by the KL-divergence from uniform distribution, and returns it as a NumPy array.

    Args:
        obs (numpy array): Observation from the environment, representing the current state.
        logits (numpy array): Predicted logits for the action probabilities at the current state.

    Returns:
        goal_congruence (float): Goal congruence score, computed as the entropy of the policy distribution over actions.
    """
    min = -1.1
    max = -0.0

    # # Convert obs and logits to PyTorch tensors
    # obs = torch.Tensor(obs).unsqueeze(0)
    # logits = torch.Tensor(logits).unsqueeze(0)

    # Compute the softmax of the logits to get the policy distribution over actions
    policy = torch.softmax(logits, dim=1)

    # Compute the entropy of the policy distribution
    entropy = -torch.sum(policy * torch.log(policy), dim=1)

    # Compute the distance from uniform distribution
    n_actions = policy.shape[1]
    uniform_policy = torch.ones_like(policy) / n_actions
    kl_divergence = torch.sum(policy * torch.log(policy / uniform_policy), dim=1)

    # Compute the goal congruence score as the negative entropy of the policy distribution
    goal_congruence = -(entropy + kl_divergence)
    # print(goal_congruence.shape)
    # Return the goal congruence score as a PyTorch tensor
    norm = NormalizeData(goal_congruence, min, max)
    return norm

# def anticipation(obs, logits):
#     """
#     Compute the anticipated next action of an RL agent in a grid world environment
#     based on the current observation and predicted logits.

#     The basic idea is to compute the expected reward for each action based on the 
#     predicted logits and the current observation, and then choose the action with 
#     the highest expected reward as the next action. This is known as the "softmax 
#     action selection" method.

#     Args:
#         obs (torch.Tensor): A tensor of shape (batch_size, height, width) representing
#             the current observation from the environment.
#         logits (torch.Tensor): A tensor of shape (batch_size, num_actions) representing
#             the predicted logits for each action.

#     Returns:
#         A tensor of shape (batch_size,) representing the anticipated next action
#         for each batch element.
#     """
#     # Compute the expected reward for each action based on the predicted logits
#     # and the current observation.
#     expected_reward = torch.sum(logits * obs.unsqueeze(1), dim=(2, 3))

#     # Apply the softmax function to obtain a probability distribution over actions.
#     action_probs = torch.softmax(expected_reward, dim=1)

#     # Choose the action with the highest expected reward as the next action.
#     anticipated_action = torch.argmax(action_probs, dim=1)

#     return anticipated_action


# def anticipation(obs, logits):
#     """
#     a simple heuristic function that assigns a score to each possible action based on the observation
#     from the environment and the predicted logits. The score can be based on various factors such as 
#     the distance to the goal, the presence of obstacles, and the predicted reward for each action.
#     """
#     # obs: a tensor of shape (batch_size, height, width, channels) representing the observation
#     # logits: a tensor of shape (batch_size, num_actions) representing the predicted logits for each action
    
#     # Compute the distance from the agent's position to the goal
#     agent_pos = torch.argmax(obs[:,:,:,0], dim=(1,2))  # find the agent's position
#     goal_pos = torch.argmax(obs[:,:,:,1], dim=(1,2))   # find the goal's position
#     distance = torch.abs(agent_pos - goal_pos)
    
#     # Compute the score for each action based on the distance to the goal and the predicted logits
#     score = logits.clone()  # start with the logits
#     score -= 0.1 * distance.unsqueeze(1)  # subtract a penalty based on the distance to the goal
    
#     # Apply a penalty for actions that lead to obstacles
#     obstacle_mask = obs[:,:,:,2] > 0.5   # identify cells with obstacles
#     obstacle_penalty = torch.zeros_like(logits).fill_(-1e9)   # a large negative penalty
#     score.masked_scatter_(obstacle_mask.unsqueeze(1), obstacle_penalty.masked_select(obstacle_mask.unsqueeze(1))))
    
#     # Select the action with the highest score
#     action = torch.argmax(score, dim=1)
    
#     return action