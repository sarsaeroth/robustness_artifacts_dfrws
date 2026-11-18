"""WL relabeling digest shared by the WL implementations.

wl_digest(own, inl, out) == sha1(f"{own}|I[{','.join(inl)}]|O[{','.join(out)}]")[:16], byte for byte,
but for high-degree nodes the signature is streamed into the hash instead of being built as one string.
A collapsed node (graph_perturb flatten) keeps its duplicate edges and a label of hundreds of KB, so
its neighbours' signatures repeat that label thousands of times: one string of several GB per node.
Neighbour lists are sorted, so repeats are adjacent and each distinct label is encoded once.
"""
import hashlib

STREAM_ABOVE = 256  # neighbours; below this the joined string is small and the direct path is faster


def _feed(h, labels):
    # comma-join labels into h; a run of equal adjacent labels is encoded once
    first = True
    have_prev = False
    prev = None
    b = None
    for lbl in labels:
        if not have_prev or lbl != prev:
            b = lbl.encode("utf-8", "ignore")
            prev = lbl
            have_prev = True
        if not first:
            h.update(b",")
        h.update(b)
        first = False


def wl_digest(own, inl, out):
    if len(inl) + len(out) < STREAM_ABOVE:
        signature = f"{own}|I[{','.join(inl)}]|O[{','.join(out)}]"
        digest = hashlib.sha1(signature.encode("utf-8", "ignore")).hexdigest()
        return digest[:16]
    h = hashlib.sha1(own.encode("utf-8", "ignore"))
    h.update(b"|I[")
    _feed(h, inl)
    h.update(b"]|O[")
    _feed(h, out)
    h.update(b"]")
    return h.hexdigest()[:16]
