import math
import numpy as np

def estimate_emotions_cpm(appraisal):
    motivational_relevance, novelty, certainty, coping_potential, anticipation, goal_congruence, accountability = appraisal
    emotions = ['Anger', 'Disgust', 'Fear', 'Guilt', 'Joy', 'Sadness', 'Shame']

    Anger = 1 / (1 + math.exp(-(1.85 * motivational_relevance - 0.91 * goal_congruence + 0.74 * coping_potential - 0.68 * accountability - 0.14)))
    Disgust = 1 / (1 + math.exp(-(0.57 * motivational_relevance - 0.45 * goal_congruence + 1.61 * coping_potential - 0.28 * accountability - 0.24)))
    Fear = 1 / (1 + math.exp(-(1.37 * motivational_relevance - 0.67 * goal_congruence - 0.17 * coping_potential + 0.52 * accountability + 0.36)))
    Guilt = 1 / (1 + math.exp(-(1.04 * motivational_relevance - 0.22 * goal_congruence - 0.45 * coping_potential - 0.61 * accountability - 0.26)))
    Joy = 1 / (1 + math.exp(-(1.04 * motivational_relevance + 1.16 * goal_congruence + 0.41 * coping_potential + 0.43 * accountability - 0.23)))
    Sadness = 1 / (1 + math.exp(-(1.69 * motivational_relevance + 0.08 * goal_congruence - 0.26 * coping_potential - 0.21 * accountability + 0.26)))
    Shame = 1 / (1 + math.exp(-(1.13 * motivational_relevance - 0.05 * goal_congruence - 0.07 * coping_potential - 0.53 * accountability - 0.05)))

    emotion_scores = [Anger, Disgust, Fear, Guilt, Joy, Sadness, Shame]
    return emotion_scores  