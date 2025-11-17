class RectangleCalculator:

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def calculateArea(self) -> float:
        return self.width * self.height