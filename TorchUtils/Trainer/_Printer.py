import sys
from math import log10
import colorama

colorama.init()
from colorama import Fore
import warnings

warnings.simplefilter("ignore")


def print_result(
    current_epoch,
    epochs,
    train_acc=None,
    train_loss=None,
    val_acc=None,
    val_loss=None,
    integer_digit=3,
    float_digit=6,
    time=None,
):
    """get_result_text

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
        time {float} -- process time (default: None)

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
    current_order = int(log10(current_epoch + 1))
    order = int(log10(epochs))
    sys.stdout.write("\r")
    result = f"Epoch [ {' '*(order-current_order)}{current_epoch+1} / {epochs} ] "
    result += (
        f"<train> accuracy: {Fore.RED}{train_acc: {integer_digit}.{float_digit}f}    {Fore.WHITE}"
        if not train_acc is None
        else ""
    )
    result += (
        f"loss: {Fore.GREEN}{train_loss: {integer_digit}.{float_digit}f}    {Fore.WHITE}"
        if not train_loss is None
        else ""
    )
    result += (
        f"training time: {Fore.BLUE}{time: {integer_digit}.{float_digit}f}  {Fore.WHITE}"
        if not time is None
        else ""
    )
    result += (
        f"| <val> accuracy: {Fore.RED}{val_acc: {integer_digit}.{float_digit}f}    {Fore.WHITE}"
        if not val_acc is None
        else ""
    )
    result += (
        f"loss: {Fore.GREEN}{val_loss: {integer_digit}.{float_digit}f}    {Fore.WHITE}"
        if not val_loss is None
        else ""
    )
    result += Fore.WHITE
    print(result)


def summarize_trainer(model, criterion, optimizer):
    """summarize_trainer

    Arguments:
    ----------
        model {nn.Module} -- model
        criterion {any} -- criterion
        optimizer {torch.optim} -- optimizer
    """
    print(Fore.RED + "\n<<< MODEL SUMMARY >>>")
    print(Fore.GREEN + "    [ Model ]" + Fore.WHITE)
    print("        " + str(model).replace("\n", "\n        "))
    print(Fore.GREEN + "\n    [ Loss function ]" + Fore.WHITE)
    print("        " + str(criterion)[:-2])
    print(Fore.GREEN + "\n    [ Optimizer ]" + Fore.WHITE)
    print("        " + str(optimizer).replace("\n", "\n        "))
    print("\n\n")


def show_progressbar(
    dataset_length, current_cnt, indicator_num=50, is_training=True, whole_time=None
):
    """show_progressbar

    Arguments:
    ----------
        dataset_length {int} -- the number of data
        current_cnt {int} -- current iteration number

    Keyword Arguments:
    ------------------
        indicator_num {int} -- the length of progress bar (default: 50)
        is_training {bool} -- True when you call this function in training loop (default: True)
    """
    indicator = (current_cnt * indicator_num) // dataset_length
    result = (
        Fore.LIGHTBLUE_EX
        + "#" * indicator
        + Fore.WHITE
        + " " * (indicator_num - indicator)
    )
    percentage = int(current_cnt / dataset_length * 100)
    percentage = 100 if percentage > 100 else percentage

    time_text = "\t\t"
    if whole_time is not None and not whole_time == 0:
        iter_per_sec = current_cnt / whole_time
        predict_remain_time = (dataset_length - current_cnt) / iter_per_sec
        predict_remain_time = predict_remain_time if predict_remain_time > 0 else 0
        time_text = "".join(
            [
                f"[{str(int(whole_time//60)).zfill(2)}:{str(int(whole_time-((whole_time//60)*60))).zfill(2)}",
                f" / {str(int(predict_remain_time//60)).zfill(2)}:{str(int(predict_remain_time-((predict_remain_time//60)*60))).zfill(2)},",
                f" {int(iter_per_sec)}it/s]",
            ]
        )

    if is_training:
        sys.stdout.write(
            f"\rTraining Progress [{result}] {percentage: 3d}%  {time_text}"
        )
    else:
        sys.stdout.write(
            f"\rValidation Progress [{result}] {percentage: 3d}%  {time_text}"
        )

    sys.stdout.flush()
