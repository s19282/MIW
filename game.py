import numpy as np


# Rock paper scissors game
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

    previous = 'R'

    opponent = "!END"

    while opponent != "END":

        if previous == 'R':
            move = np.random.choice(answers, p=probability[0])
            print(move)
        elif previous == 'P':
            move = np.random.choice(answers, p=probability[1])
            print(move)
        elif previous == 'S':
            move = np.random.choice(answers, p=probability[2])
            print(move)

        opponent = input('Enter your input:').upper()
        counter[number.get(previous)][number.get(opponent)] += 1

        previous = opponent

    print("Game ended!")


game()
