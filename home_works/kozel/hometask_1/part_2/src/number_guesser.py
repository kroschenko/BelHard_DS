import math
import sys

from number_interval import NumberInterval
from user_response import UserResponse

class NumberGuesser:
    @staticmethod
    def start():
        print("Welcome to Number guesser! You need to choose a number within the range."
              " First define the interval of numbers.")

        interval = NumberInterval()
        NumberGuesser.get_interval_side(interval, "low")
        NumberGuesser.get_interval_side(interval,"high")

        NumberGuesser.guess(interval)
        if NumberGuesser.is_continue():
            NumberGuesser.start()
        else:
            sys.exit()

    @staticmethod
    def get_number(number_position: str) -> int:
        while True:
            try:
                return int(input(f"Enter {number_position} number (positive integer): "))
            except ValueError:
                print(f"Please type {number_position} number correctly")

    @staticmethod
    def get_interval_side(interval: NumberInterval,
                          number_position: str) -> None:
        while True:
            try:
                value = NumberGuesser.get_number(number_position)
                setattr(interval, number_position, value)
                break
            except ValueError as e:
                print(e)

    @staticmethod
    def define_middle(interval: NumberInterval) -> int:
        return int(math.ceil((interval.low + interval.high) / 2))


    @staticmethod
    def guess(interval: NumberInterval) -> None:
        while True:
            try:
                middle_number = NumberGuesser.define_middle(interval)
                answer = input(
                    f"Is chosen number more or equal than {middle_number}? (Y/N) ")
                if answer.upper() == UserResponse.YES.value:
                    interval.low = middle_number
                elif answer.upper() == UserResponse.NO.value:
                    interval.high = middle_number - 1
                else:
                    print("Please type correct answer!")
                    continue
            except ValueError:
                print("It seems that you gave a wrong answer. Game is over.")

            if interval.low == interval.high:
                print(f"Chosen number - {interval.low}")
                break

    @staticmethod
    def is_continue() -> bool:
        while True:
            answer = input("Would you like to try again? (Y/N) ")
            if answer.upper() == UserResponse.YES.value:
                return True
            elif answer.upper() == UserResponse.NO.value:
                return False
            else:
                print("Please type correct answer!")
