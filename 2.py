def ugadaika():
    yes = "да"
    no = "нет"

    while True:
        print("/////Игра Угадайка/////")
        print("Загадайте число от 1 до 10 .")
        
        predpolozhenie = 1 
        schetcik = 0
        igra_zavershena = False

       
        while predpolozhenie <= 10:
            schetcik += 1

            print(f"\nЗагаданное число {predpolozhenie}? Введите '{yes}' или '{no}'")

            user_input = input()

            if user_input == yes:
                
                print(f"Ха! Я угадал число {predpolozhenie} за {schetcik} попыток.")
                igra_zavershena = True
                break
            elif user_input == no:
               
                print("Хорошо, продолжаю перебор...")
                predpolozhenie += 1
            else:
               
                print(f"Некорректный ввод. Пожалуйста, введите '{yes}' или '{no}'.")


        if not igra_zavershena:
          
            print("Кажется, вы загадали число вне диапазона 1-10 или вводили некорректные ответы.")

       
        print(f"\nХотите сыграть еще раз? Введите '{yes}' или '{no}'")
        user_choice_restart = input()

        if user_choice_restart != yes:
            print("Спасибо за игру! До свидания.")
            break