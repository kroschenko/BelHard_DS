#Угадай число
num1 = 1;
num10 = 10;
print("Загадайте число от 1 до 10")
while True:
    numx = (num1 + num10)//2
    print("Загаданное число равно ", numx, "?")
    answer = int(input("1- больше, 2 - меньше, 3 - равно "))
    if answer == 3:
        print("Угадал!")
        break
    elif answer == 1: #если число больше
         num1 = numx + 1
    elif answer == 2: #если число меньше
         num10 = numx - 1
