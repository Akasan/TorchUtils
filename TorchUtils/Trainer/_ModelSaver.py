import torch


def save_model(model, filename, is_parameter_only):
    if is_parameter_only:
        torch.save(model.state_dict(), filename)

    else:
        torch.save(model, filename)