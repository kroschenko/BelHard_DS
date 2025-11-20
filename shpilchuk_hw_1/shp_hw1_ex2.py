def game():
    print(f"Задумайте число от 1 до 10, а я его отгадаю")
    x = input(f"Ваше число больше 5 (да/нет)? ")
    if x == "да":
        for i in range(6, 10):
            x = input(f"Это число {i} (да/нет)? ")
            if x == "да":
                print(f"Ваша число: {i}")
                break
    else:
        for i in range(5, 1, -1):
            x = input(f"Это число {i} (да/нет)? ")
            if x == "да":
                print(f"Ваша число: {i}")
                break
game()