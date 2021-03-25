import numpy as np


# Rock paper scissors game
def game():
    answers = ['R', 'P', 'S']
    probability = [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]
    # RR RP RS
    # PR PP PS
    # SR SP SS

    previous = 'R'

    opponent = "!END"

    while opponent != "END":

        opponent = input('Enter your input:').upper()

        if opponent == 'R':
            print(opponent)
            # state = np.random.choice(t1, p=p_t1[0])
        elif opponent == 'P':
            print(opponent)
            # state = np.random.choice(t1, p=p_t1[1])
        elif opponent == 'S':
            print(opponent)
            # state = 4
    print("Game ended!")


game()
