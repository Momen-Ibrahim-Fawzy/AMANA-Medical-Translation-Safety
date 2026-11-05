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


@lru_cache(maxsize=1)
def _metricx(size):
    """MetricX-24 hybrid (Juraska et al., 2024), loaded as plain mT5: the score is the first decoder step's logit for
    one fixed vocabulary item, as in google-research/metricx. It is an error score from 0 (best) to 25. On a GPU the
    weights are spread over all visible devices, which the XL checkpoint needs in float32."""
    import torch
    from transformers import AutoTokenizer, MT5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained("google/mt5-large")
    kw = {"device_map": "auto"} if _device() == "cuda" else {}
    model = MT5ForConditionalGeneration.from_pretrained(f"google/metricx-24-hybrid-{size}-v2p6",
                                                        torch_dtype=torch.float32, **kw).eval()
    return tok, model


def _metricx_score(texts, size="large", bs=16):
    import torch
    tok, model = _metricx(size)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            # the reference code drops the EOS token from each input, which here means not adding it before padding
            enc = tok(texts[i:i + bs], max_length=1535, truncation=True, padding=True, add_special_tokens=False,
                      return_tensors="pt")
            ids, mask = enc.input_ids, enc.attention_mask
            dec = torch.zeros((ids.shape[0], 1), dtype=torch.long)
            logits = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                           decoder_input_ids=dec.to(model.device)).logits
            out += torch.clamp(logits[:, 0, 250089], 0, 25).float().cpu().tolist()
    return [-x for x in out]  # higher is better, like every other metric here


def metricx24(srcs, hyps, refs, size="large"):
    return _metricx_score([f"source: {s} candidate: {h} reference: {r}" for s, h, r in zip(srcs, hyps, refs)], size)


def metricx24_qe(srcs, hyps, refs, size="large"):
    return _metricx_score([f"source: {s} candidate: {h}" for s, h in zip(srcs, hyps)], size)


def metricx24_xl(srcs, hyps, refs):
    return metricx24(srcs, hyps, refs, "xl")


def metricx24_xl_qe(srcs, hyps, refs):
    return metricx24_qe(srcs, hyps, refs, "xl")


def amana_guard(srcs, hyps, refs):
    from .guard import check
    return [check(s, h).score for s, h in zip(srcs, hyps)]


def amana_guard2(srcs, hyps, refs):
    from .guard2 import check
    return [check(s, h).score for s, h in zip(srcs, hyps)]


METRICS = {
    "BLEU": bleu, "chrF": chrf, "chrF++": chrfpp, "TER (neg)": ter, "BERTScore": bertscore,
    "LaBSE-ref": labse_ref, "LaBSE-QE": labse_qe, "COMET-22": comet22, "CometKiwi-22 (QE)": cometkiwi,
    "MetricX-24": metricx24, "MetricX-24 (QE)": metricx24_qe,
    "MetricX-24 XL": metricx24_xl, "MetricX-24 XL (QE)": metricx24_xl_qe,
    "AMANA-Guard (QE)": amana_guard, "AMANA-Guard2 (QE)": amana_guard2,
}
REFERENCE_FREE = {"LaBSE-QE", "CometKiwi-22 (QE)", "MetricX-24 (QE)", "MetricX-24 XL (QE)", "AMANA-Guard (QE)", "AMANA-Guard2 (QE)",
                  "AMANA-QE"}


def contrastive_accuracy(good: np.ndarray, bad: np.ndarray) -> float:
    """ACES-style: share of pairs where the metric prefers the correct translation (ties = 0.5)."""
    good, bad = np.asarray(good), np.asarray(bad)
    return float(((good > bad) + 0.5 * (good == bad)).mean())
