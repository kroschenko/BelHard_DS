class GuessNumber:
    def __init__(self):
        print("Привет! \nПредлагаю тебе загадать число от 0 до 10, а я попробую его угадать.") # /n - переносит на новую строку

    def ask(self, question): # Спрашивает да/нет и возвращает True/False
        while True:
            answer = input(question + " (отвечай да или нет): ").strip().lower() # .strip() - удаляет пробелы покраям, .lower() - переводит регистр письма в строчные
            if answer in ("да", "нет"):
                return answer == "да"
            print("Некорректный ответ!:( \nПопробуем еще раз: отвечай пожалуйста только 'да' или 'нет'.")

    def guess(self): # Логика игры
        if self.ask("Загаданное число кратно 5?"):
            if self.ask("Это натуральное число?"):
                if self.ask("Загаданное число больше 6?"):
                    return 10
                else:
                    return 5
            else:
                return 0

        if self.ask("Загаданное число кратно 3?"):
            if self.ask("Загаданное число кратно 2?"):
                return 6
            else:
                if self.ask("Загаданное число больше 5?"):
                    return 9
                else:
                    return 3

        if self.ask("Загаданное число чётное?"):
            if self.ask("Загаданное число больше 4?"):
                if self.ask("Загаданное число больше 7?"):
                    return 8
                else:
                    return 4
            else:
                return 2
        else:
            if self.ask("Загаданное число больше 5?"):
                return 7
            else:
                return 1

    def play(self): # Первый раунд игры
        input("\nЗагадай число и нажми Enter, когда будешь готов!")
        number = self.guess()
        print(f"Я знаю! Ты загадал число {number}!")

    def run(self): # Запуск игры, повторение при желании пользователя
        while True:
            self.play()
            if not self.ask("Хочешь сыграть еще раз?"):
                print("\nХорошо, до новых встреч!")
                break

game = GuessNumber()
game.run()
