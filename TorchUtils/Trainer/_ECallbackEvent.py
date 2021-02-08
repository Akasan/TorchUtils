from enum import Enum


class ECallbackEvent(Enum):
    ITERATION = 1       # イテレーションベース
    ACCURACY = 2        # 精度ベース
    LOSS = 3            # Lossベース