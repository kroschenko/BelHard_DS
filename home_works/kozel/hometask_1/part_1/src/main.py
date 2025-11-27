from rectangle import Rectangle
from interactor import Interactor

def main():
    rectangle = Rectangle()
    Interactor.get_side(rectangle, "width")
    Interactor.get_side(rectangle, "height")

    square = Interactor.calculate_area(rectangle.width, rectangle.height)
    Interactor.print_square(square)

if __name__ == '__main__':
    main()
