"""Archive categories and standalone PNG/TGA images (no optional dependencies)."""
from i18n import tr
import struct
from pathlib import PurePosixPath

import png_util


CATEGORIES = ('All', 'Characters', 'Tracks', 'Images', 'Animations', 'Other files')


def category_match(path, category):
    path = path.replace("\\", "/").lower()
    name = PurePosixPath(path)
    characters = any(part in path for part in ("/chars/", "/charmodels/", "/trophychars/", "/dioramachars/", "/karts/"))
    tracks = any(part in path for part in ("/tracks/", "/mptracks/", "/hubworld/"))
    images = name.suffix in (".png", ".tga") or "/fonts/" in path or name.stem in ("fetextur", "introtxr", "itextur", "fonts")
    animations = name.suffix == ".igb" and ("anim" in path or "/trophychars/" in path)
    return {'All': True, 'Characters': characters, 'Tracks': tracks, 'Images': images,
            'Animations': animations, 'Other files': not (characters or tracks or images or animations)}.get(category, True)


def image(data, name):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        w, h, pixels, palette = png_util.png_decode(data, name)
        if palette is not None:
            pixels = [palette[i] for i in pixels]
        return w, h, b"".join(bytes(p) for p in pixels)
    if PurePosixPath(name).suffix.lower() != ".tga" or len(data) < 18:
        raise ValueError(tr('{0}: unrecognized PNG/TGA image', name))
    id_size, color_map, kind, first, colors, color_bits, _, _, w, h, bits, flags = struct.unpack_from("<BBBHHBHHHHBB", data)
    indexed = kind in (1, 9)
    if kind not in (1, 2, 3, 9, 10, 11) or not 0 < w * h <= 4096 * 4096 or \
            (indexed and (color_map != 1 or bits not in (8, 16) or color_bits not in (15, 16, 24, 32))) or \
            (not indexed and (color_map or bits not in (8, 24, 32))):
        raise ValueError(tr('{0}: unsupported TGA (type {1}, {2} bits)', name, kind, bits))
    if not indexed and (kind in (3, 11)) != (bits == 8):
        raise ValueError(tr('{0}: inconsistent TGA bit depth', name))
    stride, pos = bits // 8, 18 + id_size
    palette = []
    if indexed:
        entry_size = (color_bits + 7) // 8
        if pos + colors * entry_size > len(data):
            raise ValueError(tr('{0}: truncated TGA palette', name))
        for i in range(colors):
            p = data[pos + i * entry_size:pos + (i + 1) * entry_size]
            if color_bits in (15, 16):
                v = int.from_bytes(p, "little")
                rgb = tuple((((v >> shift) & 31) * 255 + 15) // 31 for shift in (10, 5, 0))
                palette.append((*rgb, 255 if color_bits == 15 or v & 0x8000 else 0))
            else:
                palette.append((p[2], p[1], p[0], p[3] if color_bits == 32 else 255))
        pos += colors * entry_size
    raw = bytearray()
    need = w * h * stride
    if kind in (1, 2, 3):
        raw = data[pos:pos + need]
    else:
        while len(raw) < need:
            if pos >= len(data):
                raise ValueError(tr('{0}: truncated TGA RLE', name))
            tag = data[pos]
            pos += 1
            count = (tag & 127) + 1
            size = stride if tag & 128 else stride * count
            packet = data[pos:pos + size]
            if len(packet) != size:
                raise ValueError(tr('{0}: truncated TGA RLE', name))
            raw += packet * count if tag & 128 else packet
            pos += size
    if len(raw) != need:
        raise ValueError(tr('{0}: truncated or oversized TGA data', name))
    out = bytearray(4 * w * h)
    for y in range(h):
        sy = y if flags & 32 else h - 1 - y
        for x in range(w):
            sx = w - 1 - x if flags & 16 else x
            off = (sy * w + sx) * stride
            p = raw[off:off + stride]
            if indexed:
                index = int.from_bytes(p, "little") - first
                if not 0 <= index < len(palette):
                    raise ValueError(tr('{0}: TGA index outside the palette', name))
                color = palette[index]
            else:
                color = (p[0], p[0], p[0], 255) if stride == 1 else (p[2], p[1], p[0], p[3] if stride == 4 else 255)
            out[4 * (y * w + x):4 * (y * w + x + 1)] = bytes(color)
    return w, h, bytes(out)
