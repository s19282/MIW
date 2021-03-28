import numpy as np


# Rock paper scissors game
def updateProbability(counter, probability):
    for i, row in enumerate(counter):
        for j, col in enumerate(row):
            probability[i][j] = counter[i][j] / np.sum(row)


def game():
    answers = ['R', 'P', 'S']
    moves = {'R': 'P', 'P': 'S', 'S': 'R'}
    number = {'R': 1, 'P': 2, 'S': 3}

    probability = [[1 / 3, 1 / 3, 1 / 3],
                   [1 / 3, 1 / 3, 1 / 3],
                   [1 / 3, 1 / 3, 1 / 3]]

    counter = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]

    # next ->, state^
    # RR RP RS
    # PR PP PS
    # SR SP SS

    previousMove = 'R'

    opponentsMove = "!END"

    while opponentsMove != "END":

        if previousMove == 'R':
            move = np.random.choice(answers, p=probability[0])
            print(move)
        elif previousMove == 'P':
            move = np.random.choice(answers, p=probability[1])
            print(move)
        elif previousMove == 'S':
            move = np.random.choice(answers, p=probability[2])
            print(move)

        opponentsMove = input('Enter your input:').upper()
        counter[number.get(previousMove)][number.get(opponentsMove)] += 1
        updateProbability(counter, probability)
        previousMove = opponentsMove

    print("Game ended!")


game()
