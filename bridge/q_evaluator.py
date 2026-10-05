"""Deployment double-Q mean; do not substitute the RL training min-Q rule."""


def mean_q(teacher, path, rgbd, goal, embodiment):
    q1, q2 = teacher.predict_pointgoal_q(path, rgbd, goal, is_target=False, embodiment=embodiment)
    return (q1 + q2) / 2
