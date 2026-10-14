
def clone_hook(module, input, output):
    if module.training:
        return output.requires_grad_(True).clone()
    else:
        return output
