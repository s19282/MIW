import numpy as np


# Rock paper scissors game
def updateProbability(counter, probability):
    for i, row in enumerate(counter):
        for j, col in enumerate(row):
            probability[i][j] = counter[i][j] / np.sum(row)


def game():
    answers = ['R', 'P', 'S']
    moves = {'R': 'P', 'P': 'S', 'S': 'R'}
    number = {'R': 0, 'P': 1, 'S': 2}

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
    previousMove = 'R'

    opponentsMove = "!END"

    while opponentsMove != "END":

        if previousMove == 'R':
            move = np.random.choice(answers, p=probability[0])
            print(moves.get(move))
        elif previousMove == 'P':
            move = np.random.choice(answers, p=probability[1])
            print(moves.get(move))
        elif previousMove == 'S':
            move = np.random.choice(answers, p=probability[2])
            print(moves.get(move))

        opponentsMove = input('Enter your input:').upper()
        counter[number.get(previousMove)][number.get(opponentsMove)] += 1
        updateProbability(counter, probability)
        previousMove = opponentsMove
        print(np.array(counter))
        print(np.array(probability))

    print("Game ended!")


game()
