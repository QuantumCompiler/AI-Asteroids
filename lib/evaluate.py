import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def evalAllTrain():
    agents = ['Agent 10', 'Agent 20', 'Agent 30', 'Agent 40', 'Agent 50']
    trainVals = [{} for _ in range(len(agents))]
    for i, agent in enumerate(agents):
        trainVals[i] = {
            'episode_num': [],
            'episode_score': [],
            'running_average': []
        }
        with open(f'../Training Results/{agent}_Train_Results.csv', 'r') as file:
            doc = csv.reader(file)
            next(doc)
            for row in doc:
                trainVals[i]['episode_num'].append(int(row[0]))
                trainVals[i]['episode_score'].append(float(row[1]))
                trainVals[i]['running_average'].append(float(row[2]))
    width = 0.25
    for i, agent in enumerate(agents):
        fig, ax = plt.subplots(figsize=(8, 5))
        episode_num = trainVals[i]['episode_num']
        episode_score = trainVals[i]['episode_score']
        running_average = trainVals[i]['running_average']
        ax.bar(episode_num, running_average, width, label='Running Average')
        ax.bar([x + width for x in episode_num], episode_score, width, label='Episode Score')
        ax.set_xlabel('Episode Number')
        ax.set_ylabel('Score')
        ax.set_title(f'Training Results for {agent}')
        ax.legend()
        plt.show()
    finalAverages = [vals['running_average'][-1] for vals in trainVals]
    agents_numbers = [10, 20, 30, 40, 50]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 5
    ax.bar(agents_numbers, finalAverages, width)
    ax.set_xlabel('Agent Number')
    ax.set_ylabel('Final Average Score')
    ax.set_title('Final Average Scores for All Agents')
    plt.show()

def evalAllTest():
    agents = ['Agent 10', 'Agent 20', 'Agent 30', 'Agent 40', 'Agent 50']
    trainVals = [{} for _ in range(len(agents))]
    evalVals = [{} for _ in range(len(agents))]
    for i, agent in enumerate(agents):
        trainVals[i] = {
            'episode_num': [],
            'episode_score': [],
            'running_average': []
        }
        with open(f'../Training Results/{agent}_Train_Results.csv', 'r') as file:
            doc = csv.reader(file)
            next(doc)
            for row in doc:
                trainVals[i]['episode_num'].append(int(row[0]))
                trainVals[i]['episode_score'].append(float(row[1]))
                trainVals[i]['running_average'].append(float(row[2]))
    for i, agent in enumerate(agents):
        evalVals[i] = {
            'episode_num': [],
            'final_score': []
        }
        with open(f'../Evaluation Results/{agent}_Eval_Results.csv', 'r') as file:
            doc = csv.reader(file)
            next(doc)
            for row in doc:
                evalVals[i]['episode_num'].append(int(row[0]))
                evalVals[i]['final_score'].append(float(row[1]))
    finalAverages = [trainVal['running_average'][-1] for trainVal in trainVals]
    finalScores = [np.mean(evalVal['final_score']) for evalVal in evalVals]
    agents_numbers = [10, 20, 30, 40, 50]
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(agents_numbers))
    ax.bar(x_pos - width / 2, finalAverages, width, label='Final Average Score')
    ax.bar(x_pos + width / 2, finalScores, width, label='Final Test Score')
    ax.set_xlabel('Agent Number')
    ax.set_ylabel('Score')
    ax.set_title('Final Average Scores vs. Final Test Scores for All Agents')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents_numbers)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    eval_typ = input("Enter the type of evaluation you'd like to perform (1 for training, 2 for evaluation): ")
    if (eval_typ == "1"):
        evalAllTrain()
    elif (eval_typ == "2"):
        evalAllTest()
    else:
        print('Invalid input. Please enter 1 or 2.')
        exit()