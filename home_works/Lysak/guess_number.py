# Игра "Угадай число"

class GuessGame:
    def play(self):
        print("Загадайте число от 1 до 10")

        for number in range(1, 11):
            answer = input(f"Это число {number}? (да/нет): ")

            if answer == "да":
                print(f"Ура! Я угадал число {number}!")
                break


# Запуск игры
game = GuessGame()
game.play()