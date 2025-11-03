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

# Build a dataset from a generator of arrays
def gen():
    for i in range(3):
        yield np.random.randn(10 + i, 4).astype(np.float32)

pm.build_map(gen(), "toy")
pm.verify_hash("toy")   # ✅ ensures data integrity

# Load individual items on demand
load, N = pm.get_loader_fn("toy")
print(f"{N} items available")
print(load(1).shape)  # → (11, 4)
```

This writes three files.
```
toy.dat          # raw binary data
toy.starts.npy   # index offsets
toy.json         # metadata + hash
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
- Use `load(i, copy=True)` to materialize a slice if you need to modify it.  
- You can safely share `.dat` files between processes (read-only).  
---

## 🌟 TL;DR

> 💾 **picomap** — simple, safe, hash-verified memory-mapped datasets.  
> No giant databases or fancy formats. Just NumPy and peace of mind.
