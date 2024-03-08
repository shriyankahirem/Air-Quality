def count_params(model):
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num_params

def model_size(model):
    """Calculate the size of the model in mb, given the number of parameters"""
    num_params = count_params(model)
    return num_params * 4 / 1024 / 1024