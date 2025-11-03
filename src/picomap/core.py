from pathlib import Path
import numpy as np
import json
import datetime
import xxhash
from tqdm.auto import tqdm
import os
from typing import Iterable


def update_hash_with_array(
    h: "xxhash.xxh3_128",
    arr: np.ndarray,
    *,
    include_meta: bool = True,
    canonical: bool = True,
    chunk_bytes: int = 1 << 20,
):
    """Incremental hasher over array."""
    a = np.asarray(arr)
    if include_meta:
        h.update(b"A|v1|")
        h.update(a.dtype.str.encode())
        h.update(np.int64(a.ndim).tobytes())
        h.update(np.asarray(a.shape, np.int64).tobytes())
    elems = max(1, chunk_bytes // max(1, a.dtype.itemsize))
    it = np.nditer(
        a,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        order=("C" if canonical else "K"),
        buffersize=elems,
    )
    for chunk in it:
        h.update(memoryview(chunk).cast("B"))
    return h


def build_map(
    generator: Iterable[np.ndarray], dest_pth: str | Path, progress: bool = True
):
    trail_dim = None
    h = xxhash.xxh3_128()

    h.update(b"D|v1|")
    idx = []
    dtype = None

    dest_pth = Path(dest_pth)
    dest_pth.parent.mkdir(parents=True, exist_ok=True)

    data_tmp = dest_pth.with_suffix(".dat.tmp")
    starts_tmp = dest_pth.with_suffix(".starts.npy.tmp")
    json_tmp = dest_pth.with_suffix(".json.tmp")

    with open(data_tmp, "wb") as f:
        for arr in tqdm(generator, disable=not progress):
            trail_dim = trail_dim or arr.shape[1:]
            dtype = dtype or arr.dtype

            if trail_dim != arr.shape[1:]:
                raise ValueError(
                    f"Trailing dimensions must be consistent across elements, found {arr.shape[1:]} instead of {trail_dim}"
                )
            if arr.dtype != dtype:
                raise ValueError(
                    f"Must have consistent dtypes across ararys, expected {dtype}, found {arr.dtype}."
                )

            a = arr if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)

            f.write(a.tobytes(order="C"))
            h = update_hash_with_array(h, a)
            idx.append(arr.shape[0])

        f.flush()
        os.fsync(f.fileno())


    nrows = int(sum(idx))

    # quick sanity check on size
    expected_size = np.dtype(dtype).itemsize * nrows * int(np.prod(trail_dim or (1,)))
    if expected_size != data_tmp.stat().st_size:
        raise IOError(f"data size mismatch: expected {expected_bytes}, got {actual_bytes}")

    h.update(b"D|end|")

    meta = dict(
        trail_dim=trail_dim,
        hash=h.hexdigest(),
        creation_time=datetime.datetime.now().strftime("%a %d %b %Y, %I:%M%p"),
        dtype=str(dtype),
        count=len(idx),
        total_rows=nrows
    )

    with open(json_tmp, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, separators=(",", ":"))

    start_pos = np.cumsum(np.concatenate([[0] + idx])).astype(np.uint64)

    with open(starts_tmp, "wb") as f:
        np.save(f, start_pos)
        f.flush()
        os.fsync(f.fileno())

    os.replace(data_tmp, data_tmp.with_suffix(""))
    os.replace(starts_tmp, starts_tmp.with_suffix(""))
    os.replace(json_tmp, json_tmp.with_suffix(""))


def verify_hash(
    mmap_pth: str | os.PathLike,
    *,
    chunk_bytes: int = 1 << 20,
) -> bool:
    """
    Recompute the dataset hash from <base>.data using the same framing as build
    (per-item A|v1| frames bounded by D|v1| ... D|end|) and compare to <base>.json.
    Raises ValueError on mismatch. Returns True on success.
    """
    base = Path(mmap_pth)
    data_p = base
    meta_p = base.with_suffix(".json")
    starts_p = base.with_suffix(".starts.npy")

    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dt = np.dtype(meta["dtype"])
    td = tuple(meta["trail_dim"])
    nrows = int(meta['total_rows'])
    starts = np.load(starts_p).astype(np.uint64, copy=False)

    if int(meta['count']) + 1 != len(starts):
        raise ValueError(f"Error: starts file has size {len(starts)}, but expected {int(meta['count']) + 1}")

    mmap = np.memmap(data_p, dtype=dt, mode="r", shape=(nrows, *td))

    h = xxhash.xxh3_128()
    h.update(b"D|v1|")
    for i in range(len(starts) - 1):
        view = mmap[starts[i] : starts[i + 1]]
        update_hash_with_array(
            h, view, include_meta=True, canonical=True, chunk_bytes=chunk_bytes
        )
    h.update(b"D|end|")

    if h.hexdigest() != meta["hash"]:
        raise ValueError("hash mismatch")
    return True


def get_loader_fn(mmap_file):
    """Returns a loader function"""
    pth = Path(mmap_file)
    idx = np.load(pth.with_suffix(".starts.npy"))
    meta = json.load(open(pth.with_suffix(".json"), "r"))

    def _load_fn(ix):
        # important to recreate the memmap each time
        memmap = np.memmap(pth, dtype=np.dtype(meta["dtype"])).reshape(
            -1, *meta["trail_dim"]
        )
        return np.array(memmap[idx[ix] : idx[ix + 1]])

    return _load_fn, len(idx) - 1
