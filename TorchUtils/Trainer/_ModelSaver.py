import torch


def save_model(
    model: torch.nn.Module, filename: str = "model.pth", is_parameter_only: bool = False
):
    """save_model

    Arguments:
    ----------
        model {torch.nn.Module} -- model

    Keyword Arguments:
    ------------------
        filename {str} -- model file name (default: "model.pth")
        is_parameter_only {bool} -- True when you want to save model with only parameters (default: True)
    """
    if is_parameter_only:
        torch.save(model.state_dict(), filename)

    else:
        torch.save(model, filename)
