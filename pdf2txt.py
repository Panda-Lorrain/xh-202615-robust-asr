#!/usr/bin/env python
# 纯标准库 PDF 文本提取（无需 poppler / pypdf）
# 用法: python pdf2txt.py <pdf路径> [最大字符数]
import zlib, re, sys

def extract(path):
    data = open(path, 'rb').read()
    streams = re.findall(rb'stream\r?\n(.*?)\r?\nendstream', data, re.DOTALL)
    out = []
    for s in streams:
        dec = None
        for w in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
            try:
                dec = zlib.decompress(s, w); break
            except Exception:
                pass
        if dec is None:
            continue
        txt = dec.decode('latin-1', errors='ignore')
        for arr in re.findall(r'\[(.*?)\]\s*TJ', txt):
            for p in re.findall(r'\(([^)]*)\)', arr):
                out.append(p)
        for m in re.findall(r'\((.*?)\)\s*Tj', txt):
            out.append(m)
        for m in re.findall(r'<([0-9A-Fa-f]+)>\s*Tj', txt):
            try:
                out.append(bytes.fromhex(m).decode('latin-1', errors='ignore'))
            except Exception:
                pass
    return out

if __name__ == '__main__':
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    pieces = extract(path)
    sys.stdout.reconfigure(encoding='utf-8')
    print(f'[文本片段数: {len(pieces)}]')
    print(' '.join(pieces)[:n])
