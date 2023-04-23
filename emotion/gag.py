import numpy as np

def circumplex_emotions(appraisal_scores):
    """
    Computes the circumplex emotions (arousal and valence) using the circumplex model of affect with the weighted average
    approach.
    :param appraisal_scores: a list or array of the six appraisal scores (motivational relevance,
                             novelty, certainty, coping potential, anticipation, goal congruence and accountability)
    :return: a tuple containing the values of arousal and valence (both between 0 and 1)
    """
    # Define the weights for each appraisal dimension based on its association with arousal and valence
    weights = np.array([[-0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25],
                        [-0.25, 0.25, -0.25, 0.25, 0.25, -0.25, 0.25]])
    
    # asc = np.array([appraisal_scores, appraisal_scores])

    weights = np.reshape(weights, (7, 2))

    # Compute the weighted average of each appraisal dimension
    weighted_appraisal = np.dot(appraisal_scores, weights)
    # print(weighted_appraisal)

    # Compute the values of arousal and valence based on the weighted average
    arousal = 0.5 + 0.5 * weighted_appraisal[0]
    valence = 0.5 + 0.5 * weighted_appraisal[1]

    return arousal, valence


def geneva_affect_grid(appraisal_scores):
    """
    Estimates emotions based on the Geneva Affect Grid, using the method described by Scherer et al. (2013).
    :param arousal: the arousal value, between 0 and 1
    :param valence: the valence (pleasure) value, between 0 and 1
    :return: a dictionary containing the estimated values of anger, disgust, fear, guilt, joy, sadness, and shame
    """
    arousal, valence = circumplex_emotions(appraisal_scores)

    # Convert the input arousal and valence values to the range [-4, 4]
    arousal = arousal * 8 - 4
    valence = valence * 8 - 4

    # Define the coordinates of the corners of the Geneva Affect Grid
    corners = np.array([[-4, 4], [-4, -4], [4, -4], [4, 4]])

    # Define the coordinates of the emotion categories on the Geneva Affect Grid
    emotions = {
        'anger': np.array([[-2, 3], [-3, 2]]),
        'disgust': np.array([[-3, -2], [-3, -3]]),
        'fear': np.array([[-2, -3], [-3, -2]]),
        'guilt': np.array([[2, 2], [1, 3]]),
        'joy': np.array([[3, 3], [1, 3]]),
        'sadness': np.array([[-2, -2], [1, -3]]),
        'shame': np.array([[2, -2], [1, -3]])
    }

    # Find the distance between the estimated emotion coordinates and the corners of the grid
    distances = np.zeros(len(emotions))
    for i, (name, coords) in enumerate(emotions.items()):
        distances[i] = np.min(np.linalg.norm(coords - np.array([arousal, valence]), axis=1))

    # Compute the weights for each emotion category based on the inverse of the distances
    weights = 1 / distances
    weights /= np.sum(weights)

    # Compute the estimated values of each emotion category as a weighted sum of the emotion category coordinates
    emotion_values = {}
    for i, (name, coords) in enumerate(emotions.items()):
        emotion_values[name] = np.sum(coords * weights[i], axis=0)

    # Convert the output emotion values to the range [0, 1]
    for name in emotion_values:
        emotion_values[name] = (emotion_values[name] + 4) / 8

    return emotion_values

