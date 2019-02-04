def adjust_learning_rate(optimizer, new_lr):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param new_lr: new learning rate
    """

    # print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    # print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
