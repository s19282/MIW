import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored


# Rock paper scissors game
def updateProbability(counter, probability):
    for i, row in enumerate(counter):
        for j, col in enumerate(row):
            probability[i][j] = counter[i][j] / np.sum(row)


def game():
    print("Rock paper scissors game")
    print("R - Rock")
    print("P - Paper")
    print("S - Scissors")
    print("END - End game and see results.")

    answers = ['R', 'P', 'S']
    responses = {'R': 'P', 'P': 'S', 'S': 'R'}
    probability = [[1 / 3, 1 / 3, 1 / 3],
                   [1 / 3, 1 / 3, 1 / 3],
                   [1 / 3, 1 / 3, 1 / 3]]
    counter = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]
    # s| next ->
    # t| RR RP RS
    # a| PR PP PS
    # t| SR SP SS
    # e|

    computerScoreList = []
    computerScore = 0
    totalGamesPlayed = 0
    previousMove = 'R'
    opponentsMove = "!END"
    plt.clf()

    while opponentsMove != "END":
        print("-------------------------")
        predictMove = ""
        if previousMove == 'R':
            predictMove = np.random.choice(answers, p=probability[0])
        elif previousMove == 'P':
            predictMove = np.random.choice(answers, p=probability[1])
        elif previousMove == 'S':
            predictMove = np.random.choice(answers, p=probability[2])

        opponentsMove = input('Enter your answer:').upper()

        if opponentsMove == "END":
            break
        elif not answers.__contains__(opponentsMove):
            print("Invalid value, try again!")
            continue
        else:
            print("Computer: ", responses.get(predictMove), " You: ", opponentsMove)
            if predictMove == opponentsMove:
                print(colored("Computer won!", "red"))
                computerScore += 1
            elif responses.get(predictMove) == opponentsMove:
                print(colored("Draw!", "blue"))
            else:
                print(colored("You won!", "green"))
                computerScore -= 1

            counter[answers.index(previousMove)][answers.index(opponentsMove)] += 1
            updateProbability(counter, probability)
            previousMove = opponentsMove
            totalGamesPlayed += 1
            computerScoreList.append(computerScore)

    print("-------------------------")
    print("Game ended!")
    print("Computer score: ", computerScore)
    print("Games played: ", totalGamesPlayed)
    plt.plot(np.arange(totalGamesPlayed), computerScoreList, 'r*')
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Computer score")
    plt.show()


game()
