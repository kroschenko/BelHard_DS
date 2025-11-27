class Rectangle:
    def __init__(self,
                 width: float | int = None,
                 height: float | int = None):
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @width.setter
    def width(self, value):
        if value <= 0:
            raise ValueError("width must be positive")
        self._width = value

    @height.setter
    def height(self, value):
        if value <= 0:
            raise ValueError("height must be positive")
        self._height = value

