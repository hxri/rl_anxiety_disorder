import numpy as np
import numpy as np
from scipy.stats import beta

def map_emotions_to_anxiety(emotions):
    ang, dis, fear, guilt, joy, sad, shame = emotions
    
    # Define the cutoff scores for state and trait anxiety
    # Values taken from:
    # Spielberger, C. D., Gorsuch, R. L., & Lushene, R. E. (1970). Manual for the state-trait anxiety inventory.
    # Consulting Psychologists Press.
    state_cutoff = 0.60
    trait_cutoff = 0.58
    
    # Calculate the total score for state and trait anxiety
    # state_score = (ang + dis + fear + guilt + sad + shame) / 6
    # trait_score = (ang + dis + fear + guilt + joy + sad + shame) / 7
    state_valence = (joy - sad + (0.5 * (ang + dis + fear + guilt + shame))) / 6
    state_arousal = (ang + dis + fear + guilt + shame) / 5
    state_score = np.sqrt(state_valence**2 + state_arousal**2)

    trait_valence = (joy - sad + (0.5 * (ang + dis + fear + guilt + shame))) / 7
    trait_arousal = (ang + dis + fear + guilt + joy + sad + shame) / 7
    trait_score = np.sqrt(trait_valence**2 + trait_arousal**2)
    # print(state_score, trait_score)
    
    # Map the scores to the presence of anxiety disorder
    if state_score >= state_cutoff or trait_score >= trait_cutoff:
        return 1
    else:
        return 0
    

def clopper_pearson(anxiety):
    num_instances = len(anxiety)
    p_anxiety = np.count_nonzero(anxiety) / num_instances
    # Calculate the proportion of positive cases
    prop_anxiety = np.mean(anxiety)

    # Calculate the 95% confidence interval using the Clopper-Pearson method
    alpha = 0.05
    ci_low, ci_high = beta.interval(1-alpha, prop_anxiety*num_instances+1, (1-prop_anxiety)*num_instances+1)

    # Determine if the confidence interval includes the threshold for clinically significant anxiety
    cut_off = 0.90
    
    # clin_cutoff = state_cutoff if p_anxiety < 0.5 else trait_cutoff  # use appropriate cutoff depending on assumed probability
    clin_sig = prop_anxiety >= cut_off

    # # Print the results
    # print(f"Proportion of positive cases: {prop_anxiety:.2f}")
    # print(f"95% confidence interval: [{ci_low:.2f}, {ci_high:.2f}]")
    # print(f"Clinically significant levels of anxiety: {clin_sig}")

    msg = f"95% confidence interval: [{ci_low:.2f}, {ci_high:.2f}] | Clinically significant levels of anxiety: {clin_sig}"
    if(clin_sig == True):
        out = 1
    else:
        out = 0
    return msg, out, prop_anxiety
