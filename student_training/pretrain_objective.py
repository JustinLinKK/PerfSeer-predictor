"""Fixed selected objective; no study imports."""
import math
import torch
from torch.nn import functional as F
BUCKET_MIB = 100000000 / 1048576
WEIGHTS = {"hard": .5, "soft": .5, "rank": .5, "rkd": .05, "upper": .2}

def memory_interval_loss(prediction, target):
    """Train a continuous value inside the desired output bucket, not at its edge."""
    import torch
    from torch.nn import functional as F
    p = prediction.float() / BUCKET_MIB
    t = target.float() / BUCKET_MIB
    low, high = ((t - 1).clamp_min(0), t)
    below, above = ((low - p).clamp_min(0), (p - high).clamp_min(0))
    outside = 2 * F.smooth_l1_loss(below, torch.zeros_like(below)) + F.smooth_l1_loss(above, torch.zeros_like(above))
    center = (low + high) / 2
    return outside + 0.05 * F.smooth_l1_loss(p, center)

def ranknet(scores, reference, kind='time', temperature=0.05):
    """Pairwise distillation: match P(i > j) implied by the reference ordering."""
    s, y = (scores.float().reshape(-1), reference.float().reshape(-1))
    i, j = torch.triu_indices(len(s), len(s), 1, device=s.device)
    if not len(i):
        return s.sum() * 0.0
    if kind == 'time':
        delta = y.clamp_min(1e-08).log()[i] - y.clamp_min(1e-08).log()[j]
        weight = torch.where(delta.abs() < math.log(1.25), 2.0, 1.0)
        target = torch.sigmoid(delta / temperature)
    else:
        delta = y[i] - y[j]
        weight = torch.ones_like(delta)
        target = torch.sigmoid(delta / (100 * temperature))
    logits = s[i] - s[j]
    return (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * weight).mean()

def rkd_distance(student, teacher):
    """Park et al. (CVPR 2019) distance-wise RKD: compare mean-normalised pairwise distances."""

    def normalised(x):
        d = torch.cdist(x.float(), x.float())
        positive = d[d > 0]
        return d / (positive.mean().clamp_min(1e-06) if positive.numel() else d.new_tensor(1.0))
    if student.size(0) < 3:
        return student.sum() * 0.0
    return F.smooth_l1_loss(normalised(student), normalised(teacher.detach()))

def teacher_bounded(student_log, teacher_log, label_log, margin=0.02):
    """Chen et al. (NeurIPS 2017): penalise label error only where the student is worse than the teacher."""
    student_error = (student_log - label_log).square()
    teacher_error = (teacher_log - label_log).square()
    return torch.where(student_error + margin > teacher_error, student_error, torch.zeros_like(student_error)).mean()

def pinball(prediction_log, label_log, quantile=0.9):
    error = label_log - prediction_log
    return torch.maximum(quantile * error, (quantile - 1) * error).mean()

def distillation_loss(student, teacher, labels=None, weights=WEIGHTS):
    """student: PerfSeerStudent output dict; teacher: TeacherTap output; labels: (B, 6) or None.

    Without labels (teacher-only transfer set, e.g. extra input-only graphs) only soft terms apply.
    """
    p, t = (student['prediction'], teacher['prediction'])
    log_t = t[:, [0, 1, 4, 5]].clamp_min(1e-06).log()
    terms = {}
    terms['soft_time'] = F.smooth_l1_loss(student['log_time'], log_t[:, :2], beta=0.05)
    terms['soft_memory'] = F.smooth_l1_loss(student['log_memory'], log_t[:, 2:], beta=0.05)
    terms['soft_busy'] = F.binary_cross_entropy_with_logits(student['busy_logit'], torch.sigmoid(teacher['busy_logit'])) + F.smooth_l1_loss(student['busy_positive'] / 100, teacher['busy_positive'] / 100, beta=0.05)
    terms['rank_time'] = (ranknet(student['log_time'][:, 0], t[:, 0]) + ranknet(student['log_time'][:, 1], t[:, 1])) / 2
    terms['rank_busy'] = ranknet(p[:, 3], t[:, 3], kind='busy')
    terms['rkd'] = rkd_distance(student['embedding'], teacher['embedding'])
    total = weights['soft'] * (terms['soft_time'] + terms['soft_memory'] + terms['soft_busy']) + weights['rank'] * (terms['rank_time'] + terms['rank_busy']) + weights['rkd'] * terms['rkd']
    if labels is not None:
        y = labels.float()
        log_y = y[:, [0, 1, 4, 5]].clamp_min(1e-06).log()
        terms['hard_time'] = F.smooth_l1_loss(student['log_time'], log_y[:, :2], beta=0.08)
        positive = (y[:, 3] > 0).float()
        terms['hard_busy'] = F.binary_cross_entropy_with_logits(student['busy_logit'], positive) + F.smooth_l1_loss(p[:, 3] / 10, y[:, 3] / 10, beta=0.5)
        terms['hard_memory'] = memory_interval_loss(p[:, 4:6], y[:, 4:6]) + teacher_bounded(student['log_memory'], log_t[:, 2:], log_y[:, 2:])
        terms['upper_memory'] = pinball(student['memory_upper'].clamp_min(1e-06).log(), log_y[:, 2:])
        total = total + weights['hard'] * (terms['hard_time'] + terms['hard_busy'] + terms['hard_memory']) + weights['upper'] * terms['upper_memory']
    return (total, {k: float(v.detach()) for k, v in terms.items()})
def loss(student, teacher, labels, reliability=None):
    return distillation_loss(student, teacher, labels)
