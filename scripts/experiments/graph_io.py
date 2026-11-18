"""Graph files (.cfg/.fcg GML) stored plain or zstd-compressed beside each other as <name>.zst.

Every reader takes the plain name ("x/sha.fcg") and gets whichever copy is on disk, the plain one
first, so callers never spell ".zst". Compressed graphs are decompressed as a stream while they
are parsed; nothing holds the whole file in memory. For a path with no .zst sibling every function
behaves like its os.path / open counterpart, so it is safe on any path.

size() is the uncompressed size, read from the zstd frame header, so scripts that order or filter
apps by graph size pick the same apps whether or not a graph is compressed.

Compress a tree with:  python scripts/graph_io.py compress DIR [DIR ...]
"""
import glob as _glob
import os
import sys
try:
    from compression import zstd  # Python 3.14+
except ImportError:  # older Python: read-only shim over the zstandard package
    import io as _io
    import zstandard as _zs

    class zstd:
        @staticmethod
        def open(p, mode="rb", encoding=None, errors=None, **_):
            r = _io.BufferedReader(_zs.ZstdDecompressor().stream_reader(open(p, "rb"), closefd=True))
            return _io.TextIOWrapper(r, encoding=encoding or "utf-8", errors=errors) if "t" in mode else r

        @staticmethod
        def get_frame_info(header):
            return type("FrameInfo", (), {"decompressed_size": _zs.get_frame_parameters(header).content_size})()

EXTS = (".cfg", ".fcg")

EXTS = (".cfg", ".fcg")


def resolve(path):
    """The on-disk copy of a graph path (plain first, else .zst), or None."""
    if os.path.isfile(path):
        return path
    z = path + ".zst"
    if os.path.isfile(z):
        return z
    return None


def exists(path):
    return resolve(path) is not None


def size(path):
    """Uncompressed size in bytes; raises FileNotFoundError like os.path.getsize."""
    r = resolve(path)
    if r is None:
        raise FileNotFoundError(path)
    if r is path:
        return os.path.getsize(r)
    # compressed copy: the size is in the zstd frame header
    with open(r, "rb") as f:
        header = f.read(18)
        return zstd.get_frame_info(header).decompressed_size


def _open(path, mode, **kw):
    # a compressor swaps x for x.zst (x.zst is complete before x is removed); retry if we lost that race
    for _ in range(3):
        r = resolve(path)
        if r is None:
            break
        try:
            if r is path:
                return open(r, mode, **kw)
            return zstd.open(r, mode, **kw)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(path)


def open_text(path, errors="replace"):
    return _open(path, "rt", errors=errors)


def open_binary(path):
    return _open(path, "rb")


def read_nx(path, **kw):
    """networkx.read_gml on either copy (read_gml accepts a binary file object)."""
    import networkx as nx
    with open_binary(path) as f:
        return nx.read_gml(f, **kw)


def glob(pattern):
    """glob() that also lists compressed graphs, under their plain names."""
    hits = set(_glob.glob(pattern))
    for p in _glob.glob(pattern + ".zst"):
        hits.add(p[:-4])  # strip ".zst"
    return sorted(hits)


def compress(path, level=9):
    """Replace a plain graph with path.zst. The .zst is written under a temporary name and renamed
    into place before the plain file is removed, so at every moment one complete copy exists."""
    tmp = path + ".zst.tmp"
    n = os.path.getsize(path)
    c = zstd.ZstdCompressor(level=level)
    c.set_pledged_input_size(n)  # puts the content size in the frame header, which size() reads
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(c.compress(chunk))
        dst.write(c.flush())
    with open(tmp, "rb") as f:
        assert zstd.get_frame_info(f.read(18)).decompressed_size == n, path
    os.replace(tmp, path + ".zst")
    os.remove(path)


def compress_tree(root, level=9):
    n = 0
    saved = 0
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(EXTS):
                p = os.path.join(d, f)
                a = os.path.getsize(p)
                compress(p, level)
                n += 1
                saved += a - os.path.getsize(p + ".zst")
                if n % 500 == 0:
                    print(f"  {root}: {n} files, {saved / 1e9:.1f} GB saved", flush=True)
    print(f"{root}: {n} files compressed, {saved / 1e9:.1f} GB saved", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "compress":
        sys.exit(__doc__)
    for root in sys.argv[2:]:
        compress_tree(root)
