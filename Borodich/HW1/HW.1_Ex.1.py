def calculate_area(a, b):
    return a * b

a = float(input("Введите длину стороны a: "))
b = float(input("Введите длину стороны b: "))

area = calculate_area(a, b)

print(f"Площадь прямоугольника равна {area:.2f}")
