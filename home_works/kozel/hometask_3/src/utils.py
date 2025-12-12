from dataclasses import dataclass
from datetime import datetime


@dataclass
class Utils:
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    PLOTS_FOLDER: str = 'plots'
    PLOT_NAME: str = 'ml_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    PLOT_EXTENSION: str = '.png'
