"""DiCoW generation.py language 强制修复(T19 critical bug, 幂等)。

背景: 对抗审查发现 DiCoW_v3_2 的 generation.py `_retrieve_init_tokens` 有死代码 bug:
  行 `if language is not None and lang_ids is not None:` 中 `lang_ids` 初始为 None 且此前
  从不赋值 → 条件恒假(死代码) → `language_to_id(zh)` 从不调用 → 落入 `detect_language`
  从音频自动检测语言。后果: enroll_infer 传的 language="zh" 被静默忽略, 退化中文音频被
  误检为英文 → 450 集中 90%(407/450)输出英文(干净音频 detect_language 检对了, 故 T17
  干净 CER=0 一直没暴露此 bug)。

修复: 把死代码条件改为 `if language is not None and hasattr(generation_config, "lang_to_id"):`
  并直接 `lang_ids = [generation_config.lang_to_id[f"<|{l}|>"] for l in languages]`(<|zh|>=50260)。
  修复后 white/pink 噪声转写从英文幻觉→正确中文; babble 仍有残留(强制只锁首位 token)。

⚠️ 此补丁打在 HF cache 的 trust_remote_code 文件上, 若 cache 被清/模型重下会丢失。
   重建环境后须重跑本脚本。验证: code/test_zh_force.py 的 A(zh) 应出中文。

用法:
  code/.venv/Scripts/python.exe code/DiCoW-inference/repro/apply_dicow_langfix.py
"""
import os

OLD = ('        if language is not None and lang_ids is not None:\n'
       '            lang_ids = [language_to_id(l) for l in languages]')
NEW = ('        if language is not None and hasattr(generation_config, "lang_to_id"):\n'
       '            # [Midea-T19 fix] 原条件 "and lang_ids is not None" 恒假(lang_ids 初始 None)=死代码,\n'
       '            # language 强制永不生效→改走 detect_language 从音频误检为英文(退化中文 90%输出英文)。\n'
       '            # 直接从 lang_to_id 取 <|zh|> token 强制首语言位。\n'
       '            lang_ids = [generation_config.lang_to_id[f"<|{l}|>"] for l in languages]')
TARGETS = [
    "E:/hf_cache/modules/transformers_modules/DiCoW_v3_2/generation.py",
    "E:/hf_cache/DiCoW_v3_2/generation.py",
]
MARKER = "Midea-T19 fix"

n_patch = n_ok = n_warn = 0
for fp in TARGETS:
    if not os.path.exists(fp):
        print(f"[skip] {fp} 不存在(模型未下载?)")
        continue
    s = open(fp, encoding="utf-8").read()
    if MARKER in s:
        print(f"[ok]   已修复  {fp}")
        n_ok += 1
        continue
    if OLD not in s:
        print(f"[WARN] 未找到目标代码块(可能已变体), 手动检查 {fp}")
        n_warn += 1
        continue
    open(fp, "w", encoding="utf-8").write(s.replace(OLD, NEW))
    print(f"[patch] 已打补丁 {fp}")
    n_patch += 1

print(f"\n[done] patch={n_patch} ok={n_ok} warn={n_warn}")
print("验证: code/.venv/Scripts/python.exe code/test_zh_force.py --n 2  (A(zh) 应出中文)")
