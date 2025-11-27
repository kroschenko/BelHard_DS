from numpy import random

class RandomNumberGame:
    def __init__(self, min_number: int, max_number: int):
        self.min_number = min_number
        self.max_number = max_number

    def start_game(self):
        random_number = random.randint(self.min_number, self.max_number)
        user_number = 0
        while user_number != random_number:
            user_number = int(input("Угадайте число от 1 до 10: "))

            if user_number > random_number:
                print("Загаданное число меньше указанного")
            elif user_number < random_number:
                print("Загаданное число больше указанного")

        print("Ура!!! Вы угадали!!!")
