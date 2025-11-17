from RandomNumberGame import RandomNumberGame

while True:
    game = RandomNumberGame(minNumber=1, maxNumber=10)
    game.startGame()

    print()
    answer = input("Хотите сыграть еще? ")
    if answer.lower() in ("yes", "y", "да"):
        continue
    else:
        break

