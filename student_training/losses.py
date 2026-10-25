from pretrain_objective import loss as pretrain_loss
def loss(student, teacher, labels, objective, reliability=None):
    if objective != "d3":
        raise ValueError("Only the D3 objective is supported")
    return pretrain_loss(student, teacher, labels)
