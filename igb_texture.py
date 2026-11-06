"""CNK Xbox's igImage formats 14/15/16: DXT1, DXT3 and DXT5.

Blocks follow Microsoft's BC1/BC2/BC3 layout (little endian, 4 by 4 texels):
https://learn.microsoft.com/en-us/windows/uwp/graphics-concepts/block-compression
The encoder favours simplicity over offline texture compression quality.
"""
import struct


def rgb565(v):
    r, g, b = v >> 11, (v >> 5) & 63, v & 31
    return (r * 255 // 31, g * 255 // 63, b * 255 // 31)


def palette(a, b, transparent):
    p, q = rgb565(a), rgb565(b)
    colors = [(*p, 255), (*q, 255)]
    if transparent and a <= b:
        colors += [(*( (x + y) // 2 for x, y in zip(p, q)), 255), (0, 0, 0, 0)]
    else:
        colors += [(*( (2 * x + y) // 3 for x, y in zip(p, q)), 255),
                   (*( (x + 2 * y) // 3 for x, y in zip(p, q)), 255)]
    return colors


def alpha_palette(a, b):
    if a > b:
        return [a, b] + [((7 - k) * a + k * b) // 7 for k in range(1, 7)]
    return [a, b] + [((5 - k) * a + k * b) // 5 for k in range(1, 5)] + [0, 255]


def decode(w, h, data, fmt):
    size = 8 if fmt == 14 else 16
    bw, bh = (w + 3) // 4, (h + 3) // 4
    if len(data) < bw * bh * size:
        raise ValueError("truncated DXT texture")
    out = bytearray(w * h * 4)
    for by in range(bh):
        for bx in range(bw):
            off = (by * bw + bx) * size
            a, b, codes = struct.unpack_from("<HHI", data, off + size - 8)
            colors = palette(a, b, fmt == 14)
            alphas = None
            if fmt == 15:
                bits = int.from_bytes(data[off:off + 8], "little")
                alphas = [((bits >> (4 * i)) & 15) * 17 for i in range(16)]
            elif fmt == 16:
                table = alpha_palette(data[off], data[off + 1])
                bits = int.from_bytes(data[off + 2:off + 8], "little")
                alphas = [table[(bits >> (3 * i)) & 7] for i in range(16)]
            for i in range(16):
                x, y = bx * 4 + i % 4, by * 4 + i // 4
                if x < w and y < h:
                    r, g, b, a = colors[(codes >> (2 * i)) & 3]
                    at = (y * w + x) * 4
                    out[at:at + 4] = bytes((r, g, b, alphas[i] if alphas else a))
    return bytes(out)


def encode(w, h, rgba, fmt):
    if len(rgba) != w * h * 4:
        raise ValueError("RGBA size does not match texture dimensions")

    def quant(c):
        r, g, b = c[:3]
        return ((r * 31 + 127) // 255 << 11) | ((g * 63 + 127) // 255 << 5) | ((b * 31 + 127) // 255)

    out = bytearray()
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            block = [tuple(rgba[4 * (min(h - 1, by + y) * w + min(w - 1, bx + x)):
                                4 * (min(h - 1, by + y) * w + min(w - 1, bx + x)) + 4])
                     for y in range(4) for x in range(4)]
            transparent = fmt == 14 and any(c[3] < 128 for c in block)
            opaque = [c for c in block if c[3] >= 128] if transparent else block
            opaque = opaque or [(0, 0, 0, 255)]
            # Choose endpoints along the block's longest RGB axis.
            axis = max(range(3), key=lambda k: max(c[k] for c in opaque) - min(c[k] for c in opaque))
            a, b = quant(max(opaque, key=lambda c: c[axis])), quant(min(opaque, key=lambda c: c[axis]))
            a, b = (min(a, b), max(a, b)) if transparent else (max(a, b), min(a, b))
            if not transparent and a == b:
                if a < 65535:
                    a += 1
                else:
                    b -= 1
            colors = palette(a, b, transparent)
            codes = 0
            for i, c in enumerate(block):
                k = 3 if transparent and c[3] < 128 else min(range(3 if transparent else 4),
                    key=lambda k: sum((c[j] - colors[k][j]) ** 2 for j in range(3)))
                codes |= k << (2 * i)
            if fmt == 15:
                bits = sum(((c[3] + 8) // 17) << (4 * i) for i, c in enumerate(block))
                out += bits.to_bytes(8, "little")
            elif fmt == 16:
                hi, lo = max(c[3] for c in block), min(c[3] for c in block)
                table = alpha_palette(hi, lo)
                bits = sum(min(range(8), key=lambda k: abs(c[3] - table[k])) << (3 * i)
                           for i, c in enumerate(block))
                out += bytes((hi, lo)) + bits.to_bytes(6, "little")
            out += struct.pack("<HHI", a, b, codes)
    return bytes(out)
