from pretrain_objective import loss as selected_loss
def loss(student, teacher, labels, objective, reliability=None):
    if objective != "d3":
        raise ValueError("Unknown published objective")
    return selected_loss(student, teacher, labels)
def task_reliability(*args):
    raise ValueError("This recipe does not use task reliability")
