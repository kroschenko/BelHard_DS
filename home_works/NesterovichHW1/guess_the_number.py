while True:
    print("Загадайте число от 1 до 10.")
    input("Нажмите Enter, когда будете готовы")

    low, high = 1, 10

    while low <= high:
        guess = (low + high) // 2
        print(f"Это число {guess}?")
        answer = input("Да / Больше / Меньше: ").strip().lower()

        if answer in ("да", "y", "yes", "д"):
            print("Угадал!")
            break
        elif answer in ("больше", ">', 'more', 'greater"):
            low = guess + 1
        elif answer in ("меньше", "<", "less"):
            high = guess - 1
        else:
            print("Не понял, ответьте: Да / Больше / Меньше")

    again = input("Хотите сыграть ещё раз? (да/нет): ").strip().lower()
    if again not in ("да", "д", "yes", "y", "+"):
        break
