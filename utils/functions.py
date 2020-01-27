import torch


def iou(pr, gt, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Source:
        https://github.com/catalyst-team/catalyst/
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union

jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score

def precision(pr, gt, threshold=0.5, activation='sigmoid'):
    
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    t = torch.sum(gt==pr).float()
    f = torch.sum(gt!=pr).float()

    return (t/(t+f))


def get_optimizer_params(model, lr, lr_weight_decay_coef, num_layers):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if lr_weight_decay_coef < 1.0:
        optimizer_grouped_parameters = [
            {'params': [
                p for n, p in param_optimizer
                if 'bert.embeddings' not in n
                and 'bert.encoder' not in n
                and not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [
                p for n, p in param_optimizer
                if 'bert.embeddings' not in n
                and 'bert.encoder' not in n
                and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [
                p for n, p in param_optimizer
                if 'bert.embeddings' in n
                and not any(nd in n for nd in no_decay)],
             'lr': lr * lr_weight_decay_coef ** (num_layers + 1), 'weight_decay': 0.01},
            {'params': [
                p for n, p in param_optimizer
                if 'bert.embeddings' in n
                and any(nd in n for nd in no_decay)],
             'lr': lr * lr_weight_decay_coef ** (num_layers + 1), 'weight_decay': 0.0}
        ]
        for i in range(num_layers):
            optimizer_grouped_parameters.append(
                {'params': [
                    p for n, p in param_optimizer
                    if 'bert.encoder.layer.{}.'.format(i) in n
                    and any(nd in n for nd in no_decay)],
                 'lr': lr * lr_weight_decay_coef ** (num_layers - i), 'weight_decay': 0.0})
            optimizer_grouped_parameters.append(
                {'params': [
                    p for n, p in param_optimizer
                    if 'bert.encoder.layer.{}.'.format(i) in n
                    and any(nd in n for nd in no_decay)],
                 'lr': lr * lr_weight_decay_coef ** (num_layers - i), 'weight_decay': 0.0})
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    return optimizer_grouped_parameters
