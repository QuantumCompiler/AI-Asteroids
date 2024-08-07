import matplotlib.pyplot as plt
import os
import csv

def evalTrain(agentName):
    episode_num = []
    episode_score = []
    running_average = []
    with open('../Training Results/'+agentName+'_Train_Results.csv', 'r') as file:
        doc = csv.reader(file)
        next(doc)
        for row in doc:
            episode_num.append(int(row[0]))
            episode_score.append(float(row[1]))
            running_average.append(float(row[2]))
    plt.plot(episode_num, episode_score, label='Episode Score')
    plt.plot(episode_num, running_average, label='Running Average')
    plt.title('Training Results For '+agentName)
    plt.xlabel('Episode Number')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    eval_typ = input("Enter the type of evaluation you'd like to perform (1 for training, 2 for evaluation): ")
    if (eval_typ == "1"):
        print('Evaluating training results...')
        evalTrain('Agent 10')
    elif (eval_typ == "2"):
        print('Evaluating agent...')
    else:
        print('Invalid input. Please enter 1 or 2.')
        exit()