"""MT evaluation metrics behind one interface: score(srcs, hyps, refs) -> list[float] (higher = better)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def bleu(srcs, hyps, refs):
    from sacrebleu.metrics import BLEU
    m = BLEU(effective_order=True)
    return [m.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]


def chrf(srcs, hyps, refs):
    from sacrebleu.metrics import CHRF
    m = CHRF()
    return [m.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]


def chrfpp(srcs, hyps, refs):
    from sacrebleu.metrics import CHRF
    m = CHRF(word_order=2)
    return [m.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]


def ter(srcs, hyps, refs):
    from sacrebleu.metrics import TER
    m = TER()
    return [-m.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]


def bertscore(srcs, hyps, refs):
    from bert_score import score
    _, _, f1 = score(list(hyps), list(refs), lang="ar", model_type="bert-base-multilingual-cased",
                     device=_device(), batch_size=64, verbose=False)
    return f1.tolist()


@lru_cache(maxsize=1)
def _labse():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/LaBSE", device=_device())


def _cos(a, b):
    m = _labse()
    ea = m.encode(list(a), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
    eb = m.encode(list(b), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
    return (ea * eb).sum(1).tolist()


def labse_ref(srcs, hyps, refs):
    return _cos(hyps, refs)


def labse_qe(srcs, hyps, refs):
    return _cos(srcs, hyps)


@lru_cache(maxsize=2)
def _comet(name):
    from comet import download_model, load_from_checkpoint
    return load_from_checkpoint(download_model(name))


def _comet_score(name, data):
    model = _comet(name)
    out = model.predict(data, batch_size=32, gpus=1 if _device() == "cuda" else 0, progress_bar=False)
    return list(map(float, out.scores))


def comet22(srcs, hyps, refs):
    return _comet_score("Unbabel/wmt22-comet-da", [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)])


def cometkiwi(srcs, hyps, refs):
    return _comet_score("Unbabel/wmt22-cometkiwi-da", [{"src": s, "mt": h} for s, h in zip(srcs, hyps)])


def amana_guard(srcs, hyps, refs):
    from .guard import check
    return [check(s, h).score for s, h in zip(srcs, hyps)]


METRICS = {
    "BLEU": bleu, "chrF": chrf, "chrF++": chrfpp, "TER (neg)": ter, "BERTScore": bertscore,
    "LaBSE-ref": labse_ref, "LaBSE-QE": labse_qe, "COMET-22": comet22, "CometKiwi-22 (QE)": cometkiwi,
    "AMANA-Guard (QE)": amana_guard,
}
REFERENCE_FREE = {"LaBSE-QE", "CometKiwi-22 (QE)", "AMANA-Guard (QE)", "AMANA-QE"}


def contrastive_accuracy(good: np.ndarray, bad: np.ndarray) -> float:
    """ACES-style: share of pairs where the metric prefers the correct translation (ties = 0.5)."""
    good, bad = np.asarray(good), np.asarray(bad)
    return float(((good > bad) + 0.5 * (good == bad)).mean())
