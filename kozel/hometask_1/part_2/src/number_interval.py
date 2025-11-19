import sys


class NumberInterval:
    def __init__(self,
                 min_value : int = 0,
                 max_value: int = sys.maxsize):
        self._high = max_value
        self._low = min_value

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @low.setter
    def low(self, value):
        if value < 0:
            raise ValueError("Number must be positive")

        if value > self.high:
            raise ValueError(f"Number must be less than {self.high}")
        self._low = value

    @high.setter
    def high(self, value):
        if value < 0:
            raise ValueError("Number must be positive")

        if value < self.low:
            raise ValueError(f"Number must be greater than {self.low}")
        self._high = value
