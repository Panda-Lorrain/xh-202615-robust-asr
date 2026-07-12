"""集成冒烟: datasetA/pos kws_0+cmd_0 跑 InferenceEngine.infer, 断言产物。

用法(必须 source setenv, 在 code/ 下跑):
  source code/setenv.sh
  cd code && .venv/Scripts/python.exe -m demo_web.selfcheck
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))  # code/

from inference_engine import InferenceEngine

ROOT = os.path.dirname(os.path.dirname(_HERE))   # 项目根
ENR = os.path.join(ROOT, "datasetA", "pos", "kws_0.wav")
REC = os.path.join(ROOT, "datasetA", "pos", "cmd_0.wav")
# 扁平命名: 被 .gitignore 的 sessions/*.wav 规则忽略, 不入库
TARGET = os.path.join(_HERE, "sessions", "selfcheck_target.wav")


def main():
    assert os.path.exists(ENR), f"缺 enrollment seed: {ENR}"
    assert os.path.exists(REC), f"缺 recognition seed: {REC}"
    eng = InferenceEngine(reject_threshold=0.0)  # thr=0 强制不拒, 验证 generate 产出(不受阈值影响)
    eng.load_models()
    r = eng.infer(ENR, REC, target_out_path=TARGET)
    show = {k: v for k, v in r.items() if k != "sims"}
    print("result:", show)
    assert "error" not in r, f"infer 出错: {r.get('error')}"
    assert r["transcript"], "transcript 为空"
    assert r["max_sim"] > 0, "max_sim<=0"
    assert os.path.exists(TARGET), "target.wav 未生成"
    print(f"\nSELFCHECK OK  transcript={r['transcript']!r}  max_sim={r['max_sim']:.3f}")


if __name__ == "__main__":
    main()
