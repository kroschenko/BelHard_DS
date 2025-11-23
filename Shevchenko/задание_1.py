length = float(input("Введите длину: "))
width = float(input("Введите ширину: "))

def rectangle(length, width):
    return (length * width)
area = rectangle(length, width)

print(f"Площадь: {area}")

if length > width:
    print("Это длинный прямоугольник.")
elif length < width:
    print("Это широкий прямоугольник.")
else:
    print("Да это же квадрат!")
    