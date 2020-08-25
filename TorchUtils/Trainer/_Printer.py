from math import log10


def get_result_text(current_epoch, epochs, train_acc=None, train_loss=None, val_acc=None, val_loss=None, integer_digit=3, float_digit=6):
    """ get_result_text

    Arguments:
    ----------
        current_epoch {int} -- current epoch
        epochs {int} -- the number of epochs

    Keyword Arguments:
    ------------------
        train_acc {float} -- training accuracy (default: None)
        train_loss {float} -- training loss (default: None)
        val_acc {float} -- validation accuracy (default: None)
        val_loss {float} -- validation loss (default: None)
        integer_digit {int} -- the number of integer digits (default: 3)
        float_digit {int} -- the number of float digits (default: 6)

    Returns:
    --------
        {str} -- result text of each epoch

    Examples:
    ---------
        >>> get_result_text(1, 100)
        Epoch [   1 / 100 ]
        >>> get_result_text(10, 100)
        Epoch [  10 / 100 ]
        >>> get_result_text(100, 100)
        Epoch [ 100 / 100 ]
        >>> get_result_text(10, 100, train_acc=0.6, train_loss=0.001234)
        Epoch [  10 / 100 ] : train acciracy: 0.600000 train loss: 0.001234
    """
    current_order = int(log10(current_epoch+1))
    order = int(log10(epochs))
    result = f"Epoch [ {' '*(order-current_order)}{current_epoch+1} / {epochs} ] => "
    result += f"train accuracy: {train_acc: {integer_digit}.{float_digit}f} " if not train_acc is None else ""
    result += f"train loss: {train_loss: {integer_digit}.{float_digit}f} " if not train_loss is None else ""
    result += f"val accuracy: {val_acc: {integer_digit}.{float_digit}f} " if not val_acc is None else ""
    result += f"val loss: {val_loss: {integer_digit}.{float_digit}f} " if not val_loss is None else ""
    return result