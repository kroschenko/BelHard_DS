from numpy import random

class RandomNumberGame:
    def __init__(self, minNumber: int, maxNumber: int):
        self.minNumber = minNumber
        self.maxNumber = maxNumber

    def startGame(self):
        randomNumber = random.randint(self.minNumber, self.maxNumber)
        userNumber = 0
        while userNumber != randomNumber:
            userNumber = int(input("Угадайте число от 1 до 10: "))

            if userNumber > randomNumber:
                print("Загаданное число меньше указанного")
            elif userNumber < randomNumber:
                print("Загаданное число больше указанного")

        print("Ура!!! Вы угадали!!!")
