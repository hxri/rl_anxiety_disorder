import math
import numpy as np

def estimate_emotions_cpm(appraisal):
    motivational_relevance, novelty, certainty, coping_potential, anticipation, goal_congruence, accountability = appraisal
    emotions = ['Anger', 'Disgust', 'Fear', 'Guilt', 'Joy', 'Sadness', 'Shame']

    Anger = 1 / (1 + math.exp(-(0.114 * motivational_relevance + 0.071 * novelty + 0.11 * certainty + 0.028 * coping_potential +  0.028 * anticipation + 0.085 * goal_congruence + 0.114 * accountability)))
    Disgust = 1 / (1 + math.exp(-(0.028 * motivational_relevance + 0.028 * novelty + 0.114 * certainty + 0.028 * coping_potential +  0.057 * anticipation + 0.028 * goal_congruence + 0.057 * accountability)))
    
    Fear = 1 / (1 + math.exp(-(0.114 * motivational_relevance + 0.085 * novelty + 0.028 * certainty + 0.028 * coping_potential +  0.114 * anticipation + 0.085 * goal_congruence + 0.085 * accountability)))
    Guilt = 1 / (1 + math.exp(-(0.114 * motivational_relevance + 0.085 * novelty + 0.114 * certainty + 0.028 * coping_potential +  0.085 * anticipation + 0.085 * goal_congruence + 0.114 * accountability)))
    
    Joy = 1 / (1 + math.exp(-(0.114 * motivational_relevance + 0.085 * novelty + 0.114 * certainty + 0.114 * coping_potential +  0.114 * anticipation + 0.114 * goal_congruence + 0.085 * accountability)))
    Sadness = 1 / (1 + math.exp(-(0.114 * motivational_relevance + 0.085 * novelty + 0.028 * certainty + 0.057 * coping_potential +  0.057 * anticipation + 0.028 * goal_congruence + 0.057 * accountability)))
    Shame = 1 / (1 + math.exp(-(0.114 * motivational_relevance + 0.085 * novelty + 0.114 * certainty + 0.028 * coping_potential +  0.085 * anticipation + 0.085 * goal_congruence + 0.114 * accountability)))

    emotion_scores = [Anger, Disgust, Fear, Guilt, Joy, Sadness, Shame]
    return emotion_scores  