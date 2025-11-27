"""Part 1"""
def get_rectangle_side(name):
    while True:
        try:
            value = float(input(f"Enter the {name} of the rectangle: "))
            if value <= 0:
                print("The side must be a positive number!")
                continue
            return value
        except ValueError:
            print("Please enter a number.")

a = get_rectangle_side("length")
b = get_rectangle_side("width")

area = a * b
print(f"Area of the rectangle: {area}")

"""Part 2"""

import random


class GuessingGame:
    def __init__(self):
        self.level = None
        self.number = None

    def run(self):
        """Main entry point."""
        self.level = self.get_level()
        self.number = random.randint(1, self.level)
        self.play()

    def get_level(self):
        """Ask user for the difficulty level."""
        while True:
            try:
                level = int(input("Enter the level (maximum number to guess): "))
                if level < 1:
                    print("Level must be a positive number!")
                    continue
                return level
            except ValueError:
                print("Please enter a valid number.")

    def get_user_guess(self):
        """Ask user for their guess."""
        while True:
            try:
                return int(input("Guess: "))
            except ValueError:
                print("Please enter a valid number.")

    def play(self):
        """Main game loop."""
        while True:
            guess = self.get_user_guess()
            if guess < self.number:
                print("Too low!")
            elif guess > self.number:
                print("Too high!")
            else:
                print("Just right!")
                break


if __name__ == "__main__":
    game = GuessingGame()
    game.run()
