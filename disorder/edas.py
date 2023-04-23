import numpy as np

def edas_anxiety(emotion_dict):
    """
    Computes the anxiety level based on the Emotional Disorders Anxiety Scale (EDAS).
    :param emotion_dict: a dictionary of emotion values over a period of time, where each key is a time point
                         and each value is a tuple of arousal and valence values for each emotion (anger, disgust,
                         fear, guilt, joy, sadness, and shame).
    :return: a float value representing the anxiety level (between 0 and 1).
    """
    # Define the weights for each emotion based on the EDAS model
    weights = {'anger': 0.15, 'disgust': 0.10, 'fear': 0.25, 'guilt': 0.15, 'joy': -0.10, 'sadness': 0.30, 'shame': 0.15}

    # Compute the weighted average of the emotion values over the given period of time
    weighted_emotion_sum = 0
    weight_sum = 0
    for emotion, emotion_values in emotion_dict.items():
        arousal, valence = emotion_values
        weighted_emotion_sum += arousal * valence * weights[emotion]
        weight_sum += abs(weights[emotion])
    weighted_average = weighted_emotion_sum / weight_sum

    # Compute the anxiety level based on the weighted average of the emotion values
    anxiety_level = (1 - weighted_average) / 2

    # Check if the anxiety level is clinically significant and print a message
    if anxiety_level >= 0.3:
        print("Clinically significant level: {}" .format(anxiety_level))
    else:
        print("No clinically significant level of anxiety relevant to anxiety disorder is present.")
    return anxiety_level
