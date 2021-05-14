class NotExistLayerNameError(Exception):
    def __init__(self, mes):
        super(NotExistLayerNameError, self).__init__(mes)


class NotTensorError(Exception):
    def __init__(self, mes):
        super(NotTensorError, self).__init__(mes)


class InvalidShapeError(Exception):
    def __init__(self, mes=""):
        super(InvalidShapeError, self).__init__(mes)
