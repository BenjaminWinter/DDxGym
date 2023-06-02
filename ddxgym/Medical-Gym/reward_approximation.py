import gym
import sys
import csv
import json
import statistics
from tabulate import tabulate

env = gym.make("gym_medical:doctorsim-v0", data_path="data/csv/", max_diseases = int(sys.argv[1]) if len(sys.argv) > 1 else None, is_csv=True)

diseases = {}
lengths  = []

with open("../Medical-RL/flask/static/episode_log.csv", 'r') as csvfile:
    datareader = csv.DictReader(csvfile, delimiter=";")
    for row in datareader:
        row["procedures"] = row["procedures"].split(",")
        row["procedures"] = [x.replace("'", "").strip() for x in row["procedures"]]
        
        lengths.append(int(row["steps"]))
        c_rewards = []
        for _ in range(5000):
            list_actions = row["procedures"].copy()
            c_reward = 0
            env.reset_with_disease(row["disease"])
            env._has_reset = True

            done = False
            while not (done or len(list_actions) < 1):
                a = env.procedures_to_actions.index(list_actions.pop(0))
                obs, r, done, info = env.step(a)
                c_reward += r
            c_rewards.append(c_reward)
        diseases[row["disease"]] = statistics.mean(c_rewards)
        
print(f"length mean: {statistics.mean(lengths)}")
print(diseases)
        
                                              



# while 1==1:
#     obs = env.reset()
#     assert env.observation_space.contains(obs)
#     #print(f"Obs: {env.tokenizer.decode(obs, skip_special_tokens=True)}")
#     done = False
#     while not done:
#         env.render()
#         a = -3
#         while a < 0:
#             a = int(input("Enter Action Nr.: "))
#             if a == -1:
#                 d = env.get_patient().disease
#                 table = []
#                 for s in d.symptoms.values():
#                     s = s["symptom"]
#                     symptom = s.name
#                     examinations = ", ".join(f"{e.name}({env.procedures_to_actions.index(e.id)})" for e in s.examinations)
#                     treatments = ", ".join(f"{t.name}({env.procedures_to_actions.index(t.id)})" for t in s.treatments)
#                     is_main = s.is_main
#                     table += [[ symptom, examinations, treatments, is_main]]
#                 for i, p in enumerate(env.procedures_to_actions):
#                     print(f"{i}  {p}")
#                 print(tabulate(table, headers=["Symptom", "Examinations", "Treatments", "Is Main Symptom"]))
#         obs, r , done, info = env.step(a)
#         assert env.observation_space.contains(obs)
#         print(f"Reward: {r} | done: {done}")
#         #print(f"Obs: {env.tokenizer.decode(obs, skip_special_tokens=True)}")
#         print(obs)
#         print(obs.shape)
