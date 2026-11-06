"""PNG in and out with the standard library, and a palette for a picture (median cut): what the texture
import and export need. (Copied from the Crash WoC editor's gsc_tex.py, so that this tool stands on its own.)"""
from __future__ import annotations

import struct
import zlib


class PngError(ValueError):
    pass


def png_bytes(w: int, h: int, pixels: bytes, palette: list | None = None) -> bytes:
    """RGBA (palette None) or 8-bit indexed with the palette's alpha as tRNS."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    stride = w if palette else w * 4
    rows = b"".join(b"\0" + pixels[y * stride:(y + 1) * stride] for y in range(h))
    head = struct.pack(">IIBBBBB", w, h, 8, 3 if palette else 6, 0, 0, 0)
    body = chunk(b"IHDR", head)
    if palette:
        body += chunk(b"PLTE", b"".join(bytes(c[:3]) for c in palette))
        body += chunk(b"tRNS", bytes(c[3] for c in palette))
    return b"\x89PNG\r\n\x1a\n" + body + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b"")


def png_decode(data: bytes, name: str = "PNG") -> tuple[int, int, list, list | None]:
    """(width, height, pixels, palette): indices and the palette's RGBA for an indexed PNG, else RGBA tuples."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise PngError(f"{name}: not a PNG")
    pos, idat, plte, trns, head = 8, bytearray(), None, b"", None
    while pos < len(data):
        n, tag = struct.unpack_from(">I4s", data, pos)
        body = data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            head = struct.unpack(">IIBBBBB", body)
        elif tag == b"PLTE":
            plte = body
        elif tag == b"tRNS":
            trns = body
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    if head is None:
        raise PngError(f"{name}: no IHDR")
    w, h, depth, ctype, _comp, _filt, interlace = head
    if interlace or depth == 16 or ctype not in (0, 2, 3, 4, 6) or ctype in (2, 4, 6) and depth != 8:
        raise PngError(f"{name}: PNG colour type {ctype}, depth {depth}, interlace {interlace}: save it as an "
                       "8-bit RGBA or an indexed PNG without interlacing")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    bpp = max(1, channels * depth // 8)
    stride = (w * channels * depth + 7) // 8
    raw = zlib.decompress(bytes(idat))
    rows, prev, p = [], bytearray(stride), 0
    for _ in range(h):
        ftype, line = raw[p], bytearray(raw[p + 1:p + 1 + stride])
        p += 1 + stride
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b, c = prev[i], prev[i - bpp] if i >= bpp else 0
            if ftype == 1:
                line[i] = (line[i] + a) & 255
            elif ftype == 2:
                line[i] = (line[i] + b) & 255
            elif ftype == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif ftype == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line
    if depth < 8:
        per = 8 // depth
        samples = [[row[x // per] >> (8 - depth * (x % per + 1)) & (1 << depth) - 1 for x in range(w)] for row in rows]
    else:
        samples = [list(row) for row in rows]
    if ctype == 3:
        if plte is None:
            raise PngError(f"{name}: indexed PNG without PLTE")
        palette = [(plte[3 * i], plte[3 * i + 1], plte[3 * i + 2], trns[i] if i < len(trns) else 255)
                   for i in range(len(plte) // 3)]
        return w, h, [v for row in samples for v in row], palette
    scale = 255 // ((1 << depth) - 1)
    pixels = []
    for row in samples:
        for x in range(w):
            s = row[x * channels:(x + 1) * channels]
            if ctype == 0:
                pixels.append((s[0] * scale,) * 3 + (255,))
            elif ctype == 4:
                pixels.append((s[0],) * 3 + (s[1],))
            elif ctype == 2:
                pixels.append(tuple(s) + (255,))
            else:
                pixels.append(tuple(s))
    return w, h, pixels, None


def quantise(pixels: list, n: int) -> tuple[list, list]:
    """A palette of at most n colours and an index per pixel: the colours as they are when they fit, else median
    cut over r, g, b, a (of the colours rounded to 5 bits, weighted by their texels) and the nearest colour."""
    counts: dict = {}
    for p in pixels:
        counts[p] = counts.get(p, 0) + 1
    if len(counts) <= n:
        palette = sorted(counts)
        index = {c: i for i, c in enumerate(palette)}
        return palette, [index[p] for p in pixels]
    groups: dict = {}  # a 5-bit colour: [texels, sum r, g, b, a]
    for c, k in counts.items():
        g = groups.setdefault(tuple(v >> 3 for v in c), [0, 0, 0, 0, 0])
        g[0] += k
        for i in range(4):
            g[i + 1] += c[i] * k
    cols = [(tuple((g[i + 1] + g[0] // 2) // g[0] for i in range(4)), g[0]) for g in groups.values()]

    def spread(box):
        """The channel a box spreads most over, and by how much."""
        ranges = [max(c[k] for c, _ in box) - min(c[k] for c, _ in box) for k in range(4)]
        return max(range(4), key=ranges.__getitem__), max(ranges)

    boxes = [(cols, *spread(cols))]
    while len(boxes) < n:
        i = max(range(len(boxes)), key=lambda i: boxes[i][2] if len(boxes[i][0]) > 1 else -1)
        box, k, size = boxes[i]
        if len(box) < 2 or size == 0:
            break
        box = sorted(box, key=lambda c: c[0][k])
        total, acc, half = sum(w for _, w in box), 0, 1
        for half, (_, w) in enumerate(box, 1):
            acc += w
            if acc * 2 >= total:
                break
        half = min(max(half, 1), len(box) - 1)
        boxes[i] = (box[:half], *spread(box[:half]))
        boxes.append((box[half:], *spread(box[half:])))
    palette = []
    for box, _, _ in boxes:
        weight = sum(w for _, w in box)
        palette.append(tuple((sum(c[k] * w for c, w in box) + weight // 2) // weight for k in range(4)))
    return palette, nearest(palette, pixels)


def nearest(palette: list, colours: list) -> list[int]:
    """The palette entry nearest each colour; colours that round to the same 5-bit colour share the search."""
    exact = {c: i for i, c in enumerate(palette)}
    cache: dict = {}
    out = []
    for c in colours:
        i = exact.get(c)
        if i is None:
            key = (c[0] >> 3, c[1] >> 3, c[2] >> 3, c[3] >> 3)
            i = cache.get(key)
            if i is None:
                r, g, b, a = c
                i = cache[key] = min(range(len(palette)), key=lambda j: (r - palette[j][0]) ** 2 +
                                     (g - palette[j][1]) ** 2 + (b - palette[j][2]) ** 2 + (a - palette[j][3]) ** 2)
        out.append(i)
    return out
