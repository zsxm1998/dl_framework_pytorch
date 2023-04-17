import torch

def load_param(model, model_weight, map_location='cpu', logger=None):
    if logger is None:
        print_func = print
    else:
        print_func = logger.info
    model_state_dict = model.state_dict()
    if isinstance(model_weight, str):
        loaded_state_dict = torch.load(model_weight, map_location=map_location)
    else:
        loaded_state_dict = model_weight
    for key in loaded_state_dict:
        if key not in model_state_dict or model_state_dict[key].shape != loaded_state_dict[key].shape:
            print_func(f'load_param: ignore {key}')
            continue
        model_state_dict[key].copy_(loaded_state_dict[key])