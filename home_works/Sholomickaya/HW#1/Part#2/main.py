import time
def guess_number():

    while True:
        print("=" * 20)
        print("Загадайте любое число от 1 до 10")
        print("=" * 20)
        time.sleep(2)
        low = 1
        high = 10

        while True:
            number = (low + high) // 2

            user_response = input(f"Вы загодали число {number}? \n"
                                  f"Если я угадал - нажмите 1\n"
                                  f"Если ваше число больше - нажмите 2\n"
                                  f"Если ваше число меньше - нажмите 3   \n")

            if user_response == "1" :
                print("=" * 20)
                print(f"УРА! Я ВЫИГРАЛ!\n")
                break
            elif user_response == "2" :
                low = number + 1
            elif user_response == "3" :
                high = number - 1
            else:
                print("=" * 20)
                print("Вы ввели неверное число!")
                continue

            if low > high:
                print("Вы где-то ошиблись, нужно начать игру заново")
                break

        while True:
            print("=" * 20)

            play_again = input("Хотите сыграть еще раз?\n"
                                   "Да - нажмите 1\n"
                                   "Нет - нажмите 2   \n" )

            if play_again == "1":
                break
            elif play_again == "2":
                print("До свидания!")
                return
            else:
                print("Вы ввели неверное значение!")


guess_number()