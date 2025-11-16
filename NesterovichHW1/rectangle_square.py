try:
    side1 = float(input('Введите первую сторону прямоугольника: '))
    side2 = float(input('Введите вторую сторону прямоугольника: '))
except ValueError:
    print("Ошибка: нужно вводить числа")
else:
    if side1 <= 0 or side2 <= 0:
        print("Вы ввели неверные значения сторон")
    else:
        print(side1 * side2)
