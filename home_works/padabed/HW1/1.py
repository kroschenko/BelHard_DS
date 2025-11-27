from RectangleCalculator import RectangleCalculator

width = float(input("Введите ширину прямоугольника: "))
height = float(input("Введите длину прямоугольника: "))

calc = RectangleCalculator(width, height)

print(calc.calculate_area())
