class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def getArea(self):
        return self.width * self.height

    @staticmethod
    def calculateArea():
        """Calculates the area an arbitrary rectangle"""

        width = float(input("Enter the width of the rectangle: "))
        height = float(input("Enter the height of the rectangle: "))

        return f"The area of the rectangle is {width * height}"


