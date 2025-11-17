def ugadaika():
    yes = "да"
    no = "нет"

    print("/////Игра Угадайка/////")
    print("Загадайте число от 1 до 10 .")

    predpolozhenie = 1 
    schetcik = 0

       while predpolozhenie <= 10:
        schetcik += 1

        print(f"\nЗагаданное число {predpolozhenie}? Введите '{yes}' или {no}")

        user_input = input()

        if user_input == yes:

            print(f"Ха! Я угадал число {predpolozhenie} за {schetcik} попыток.")
            break
        else:

            print("Хорошо, продолжаю перебор...")
            predpolozhenie += 1
    else:
        print("Кажется, вы загадали число вне диапазона 1-10 или вводили некорректные ответы.")
ugadaika()

print(f"\nХотите сыграть еще раз? Введите '{yes}' или '{no}'")
