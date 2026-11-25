import struct, numpy as np
def load(p):
    out = []
    with open(p, "rb") as f:
        while True:
            h = f.read(8)
            if len(h) < 8: break
            k, n = struct.unpack("ii", h)
            b = np.frombuffer(f.read(20 * n), np.float32).reshape(n, 5)
            e = np.frombuffer(f.read(2048 * n), np.float32).reshape(n, 512)
            out.append((b, e))
    return out
a, b = load(r"E:\bench\results\cpp\dump_fp16.bin"), load(r"E:\bench\results\cpp\dump_int8.bin")
cos = []
for (ba, ea), (bb, eb) in zip(a, b):
    x = ea / np.linalg.norm(ea, axis=1, keepdims=True); y = eb / np.linalg.norm(eb, axis=1, keepdims=True)
    cos += (x * y).sum(1).tolist()
E1 = np.concatenate([e for _, e in a]); E2 = np.concatenate([e for _, e in b])
E1 /= np.linalg.norm(E1, axis=1, keepdims=True); E2 /= np.linalg.norm(E2, axis=1, keepdims=True)
s1, s2 = E1 @ E1.T, E2 @ E2.T
np.fill_diagonal(s1, -1); np.fill_diagonal(s2, -1)
c = np.array(cos)
print(f"INT8 vs FP16 embeddings: cosine mean {c.mean():.4f} p5 {np.percentile(c, 5):.4f} min {c.min():.4f} n={len(c)}; same nearest neighbour {100 * (s1.argmax(1) == s2.argmax(1)).mean():.1f}%")
