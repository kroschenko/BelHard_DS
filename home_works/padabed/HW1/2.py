from RandomNumberGame import RandomNumberGame

while True:
    game = RandomNumberGame(min_number=1, max_number=10)
    game.start_game()

    print()
    answer = input("Хотите сыграть еще? ")
    if answer.lower() in ("yes", "y", "да"):
        continue
    else:
        break
