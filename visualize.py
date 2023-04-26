import argparse
import time
import numpy as np
import torch
import os
import csv

import gym_minigrid
import utils
from emotion import estimate_emotions_cpm, geneva_affect_grid
from disorder import map_emotions_to_anxiety, clopper_pearson, edas_anxiety
import matplotlib.pyplot as plt

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--appraisal", action="store_true", default=False,
                    help="internal appraisal model is used in the action decision")
parser.add_argument("--emotion", action="store_true", default=False,
                    help="Estimate the emotion from appraisals")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=args.argmax, use_text=args.text)
print("Agent loaded\n")

# Run the agent

if args.gif:
   from array2gif import write_gif
   frames = []

# Create results folder if not exist
if not os.path.exists("storage/" + args.model + "/results/"):
    os.makedirs("storage/" + args.model + "/results/")

# Create a window to view the environment
env.render('human')
anxiety_per_episode = []
for episode in range(args.episodes):
    obs = env.reset()

    # Set the initial values of the input variables at t=0
    appraisal = [[], [], [], [], [], [], []]
    dist = None
    accountable = torch.ones(size=(1,1)) 
    anxiety = []
    with open("storage/" + args.model + "/results/{}.txt" .format(episode), "w") as outfile:
        while True:
            env.render('human', appraisal=appraisal)
            if args.gif:
                frames.append(np.moveaxis(env.render("rgb_array"), 2, 0))

            dist, action, appraisal = agent.get_action(obs, dist, appraisal, accountable)
            emotions = estimate_emotions_cpm(np.array(appraisal)[:,-1])
            anx = map_emotions_to_anxiety(emotions)
            anxiety.append(anx)

            outfile.write("Emotion : \n" + str(emotions) + "\n")

            app = np.array(appraisal)[:,-1]
            outfile.write("Appraisal : \n")
            for a in app:
                outfile.write(str(float(a)) + ' ')
            outfile.write('\n\n')

            obs, reward, done, _, accountable = env.step(action)
            accountable = torch.Tensor([accountable]).reshape(1,1)
            agent.analyze_feedback(reward, done)

            if done or env.window.closed:
                # outfile.close()
                time.sleep(0.5)
                break
        if env.window.closed:
            outfile.close()
            break
        
        # fig = plt.figure(figsize=(6, 3))
        # plt.plot(anxiety)
        # fig.savefig("storage/" + args.model + "/results/{}_plot.png" .format(episode))
        # plt.close()
        
        msg, dat, aval = clopper_pearson(anxiety)
        print(msg)
        anxiety_per_episode.append(dat)
        outfile.write('\n\n' + msg)
        outfile.close()
    # print(anxiety)
    # print("\n")
    # if(np.count_nonzero(anxiety == 1) >= 1):
    #     print("Agent has expereinced anxiety")
    # else:
    #     print("Agent is anxiety free")
    # print("End of episode {}" .format(episode))
with open("storage/" + args.model + "/results/anxiety.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(anxiety_per_episode)

if args.gif:
    print("Saving gif... ", end="")
    write_gif(np.array(frames), "storage/" + args.model + "/results/" + args.gif+".gif", fps=1/args.pause)
    print("Done.")
