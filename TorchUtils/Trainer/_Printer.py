from math import log10


def get_result_text(current_epoch, epochs, train_acc=None, train_loss=None, val_acc=None, val_loss=None, integer_digit=3, float_digit=6):
    current_order = int(log10(current_epoch+1))
    order = int(log10(epochs))
    result = f"Epoch [{' '*(order-current_order)}{current_epoch+1} / {epochs} ] : "
    result += f"train accuracy: {train_acc: {integer_digit}.{float_digit}f} " if not train_acc is None else ""
    result += f"train loss: {train_loss: {integer_digit}.{float_digit}f} " if not train_loss is None else ""
    result += f"val accuracy: {val_acc: {integer_digit}.{float_digit}f} " if not val_acc is None else ""
    result += f"val loss: {val_loss: {integer_digit}.{float_digit}f} " if not val_loss is None else ""
    return result