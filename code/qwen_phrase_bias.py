"""Qwen 解码期短语偏置：只提升已进入声学 top-K 的候选 token。

这不是 hard grammar，也不会禁止普通中文输出。短语的首 token 或匹配前缀后的
下一 token 只有在当前 logits top-K 内时才获得小幅加分，从而降低纯热词 prompt
在低信息音频上凭空回吐词表的风险。
"""
import json

import torch


def load_phrases(path):
    """读取 UTF-8 JSON 字符串数组或逐行短语文件，去空、稳定去重。"""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError("phrase file must be a JSON string array or one phrase per line")
    return list(dict.fromkeys(x.strip() for x in value if x.strip()))


def tokenize_phrases(tokenizer, phrases):
    """短语转 token id；过滤空编码并稳定去重。"""
    tokenized = []
    seen = set()
    for phrase in phrases:
        ids = tuple(tokenizer.encode(phrase, add_special_tokens=False))
        if ids and ids not in seen:
            seen.add(ids)
            tokenized.append(ids)
    return tokenized


class AcousticTopKPhraseBias:
    """Transformers-compatible logits processor for softly biased phrases."""

    def __init__(self, phrase_token_ids, bias=0.8, top_k=20):
        if bias <= 0:
            raise ValueError("bias must be > 0")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        self.phrases = [tuple(int(t) for t in phrase) for phrase in phrase_token_ids]
        if not self.phrases or any(not phrase for phrase in self.phrases):
            raise ValueError("at least one non-empty tokenized phrase is required")
        self.bias = float(bias)
        self.top_k = int(top_k)

    @staticmethod
    def _next_candidates(sequence, phrases):
        """返回可开始短语或延续当前短语前缀的 token ids。"""
        candidates = {phrase[0] for phrase in phrases}
        for phrase in phrases:
            max_prefix = min(len(sequence), len(phrase) - 1)
            for prefix_len in range(max_prefix, 0, -1):
                if tuple(sequence[-prefix_len:]) == phrase[:prefix_len]:
                    candidates.add(phrase[prefix_len])
                    break
        return candidates

    def __call__(self, input_ids, scores):
        k = min(self.top_k, scores.shape[-1])
        top_ids = torch.topk(scores, k=k, dim=-1).indices
        for batch_index in range(scores.shape[0]):
            allowed = set(int(x) for x in top_ids[batch_index].tolist())
            sequence = [int(x) for x in input_ids[batch_index].tolist()]
            candidates = self._next_candidates(sequence, self.phrases)
            selected = list(candidates.intersection(allowed))
            if selected:
                scores[batch_index, selected] += self.bias
        return scores
