# picomap  
`picomap` makes it easy to store and load datasets for machine learning. It is *tiny* (<200 LOC) but works well whenever I have a non-standard dataset and want efficient loading. 

---
## 🚀 Why picomap?

<div align="center">
  <img width="384" alt="picomap_vs_others" src="https://github.com/user-attachments/assets/ff1b9735-73d6-47a0-b106-51203e6effb4" />
  <p style="font-size: 0.9em; color: #666; margin-top: 0.5em;">
    <em>Actual photo of modern dataset solutions vs picomap.</em>
  </p></div>


✅ **Fast** — writes arrays directly to disk in binary form  
✅ **Reproducible** — per-item hashing for content verification  
✅ **Simple** — one Python file, only dependencies are `numpy`, `xxhash`, and `tqdm`. Tbh you probably don't need the last two.

---

## 🧩 Installation

```bash
pip install picomap
```
---

## 💡 Quick Example

```python
import numpy as np
import picomap as pm

# Build a ragged dataset from a generator of arrays
lens = np.random.randint(16, 302, size=(101,))
arrs = [np.random.randn(l, 4, 16, 3) for l in lens]

pm.build_map(arrs, "ds/test")
assert pm.verify_hash("ds/test.dat")

# Load individual items on demand
load, N = pm.get_loader_fn("ds/test.dat")
for i in range(N):
  assert np.allclose(arrs[i], load(i))
```

This writes three files and creates the directory `ds`
```
ds/test.dat          # raw binary data
ds/test.starts.npy   # index offsets
ds/test.json         # metadata + hash
```

---

## ⚙️ API Summary

| Function | Purpose |
|-----------|----------|
| `build_map(gen, path)` | Stream arrays → build dataset on disk |
| `verify_hash(path)` | Recompute & validate hash |
| `get_loader_fn(path)` | Return `(loader_fn, count)` for random access |
| `update_hash_with_array(h, arr)` | Internal helper (streamed hashing) |

---

## 🧰 Tips
- All arrays must share the same dtype and trailing dimensions.
- The first dimension can be ragged across the dataset (i.e., you can have sequences with shapes `(*, d1, d2, ..., dn)`).
- Use `load(i, copy=True)` to materialize a slice if you need to modify it. I generally copy the tensor to GPU in a training loop anyway.