class Interactor:
    @staticmethod
    def calculate_area(width: float | int,
                       height: float | int) -> float:
        return width * height

    @staticmethod
    def get_side(figure: object,
                 side_name: str) -> None:
        while True:
            try:
                value = Interactor.get_value(side_name)
                setattr(figure, side_name, value)
                break
            except ValueError as e:
                print(e)

    @staticmethod
    def print_square(square: float) -> None:
        print(f"The square is: {square}")

    @staticmethod
    def get_value(value_name: str) -> float:
        while True:
            try:
                return float(input(f"Enter {value_name}: "))
            except ValueError:
                print(f"Please type {value_name} correctly")
