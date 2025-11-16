def get_positive_number(prompt):
    #Функция для получения положительного числа от пользователz
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
            else:
                print("Запрещен ввод отрицательных чисел! Попробуйте снова.")
        except ValueError:
            print("Ошибка: введите корректное число!")

def calculate_area(length, width):
 #Функция для расчета площади прямоугольника      
  return length * width

def main():
    print("Калькулятор площади прямоугольника")
    
    # Ввод данных
    length = get_positive_number("Введите длину прямоугольника: ")
    width = get_positive_number("Введите ширину прямоугольника: ")
    
    # Расчет
    area = calculate_area(length, width)
    
    # Вывод результата
    print(f"\nРезультат:")
    print(f"Длина: {length}")
    print(f"Ширина: {width}")
    print(f"Площадь: {area}")

# Запуск программы
if __name__ == "__main__":
    main()

