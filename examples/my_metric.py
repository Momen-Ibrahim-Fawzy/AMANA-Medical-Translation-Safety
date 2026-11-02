"""A small example for `python -m amana benchmark examples.my_metric:char_overlap`. Replace it with your metric."""


def char_overlap(sources, hypotheses, references):
    return [len(set(h) & set(r)) / max(len(set(h) | set(r)), 1) for h, r in zip(hypotheses, references)]
