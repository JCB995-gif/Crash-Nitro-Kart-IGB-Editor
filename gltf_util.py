"""glTF 2.0 pieces: a buffer of accessors to write, a .gltf/.glb read back with its accessors, matrices (16 floats
row by row, a point times the matrix: flattened as they are, they are glTF's column-major ones) as translation,
rotation and scale and back, an animation sampler sampled at a time. (Copied from the Crash WoC editor's
cut_bones.py, so that this tool stands on its own.)"""
from __future__ import annotations

import base64
import bisect
import json
import math
import struct
from pathlib import Path


class GltfError(ValueError):
    pass


def part3(m: list) -> list:
    return [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]


def full(r: list, t: list) -> list:
    return [r[0], r[1], r[2], 0.0, r[3], r[4], r[5], 0.0, r[6], r[7], r[8], 0.0, t[0], t[1], t[2], 1.0]


def quat_of(r: list) -> list:
    """The quaternion of a row-major 3x3 rotation (rows the axes)."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        q = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        q = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        q = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return [x / n for x in q]


def rot_of(q: list) -> list:
    """The row-major 3x3 (rows the axes) of a quaternion."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    m = [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
         2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]


def cross(a: list, b: list) -> list:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def scale_rotation(m3: list, fallback: list | None = None) -> tuple[list, list]:
    """A 3x3 as its row scales and rotation; a row scaled to nothing (a joint shrunk away) takes its direction from
    the other rows, or from fallback's rotation when there are none."""
    rows = [m3[0:3], m3[3:6], m3[6:9]]
    s = [math.sqrt(sum(v * v for v in row)) for row in rows]
    eps = 1e-8
    unit = [[v / s[k] for v in row] if s[k] > eps else None for k, row in enumerate(rows)]
    ok = [u is not None for u in unit]
    if sum(ok) == 0:
        r = list(fallback or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        return [0.0, 0.0, 0.0], r
    if sum(ok) == 1:
        k = ok.index(True)
        a = unit[k]
        helper = [0.0, 1.0, 0.0] if abs(a[1]) < 0.9 else [1.0, 0.0, 0.0]
        b = cross(helper, a) if k != 1 else cross(a, helper)
        n = math.sqrt(sum(v * v for v in b))
        unit[(k + 1) % 3] = [v / n for v in b]
        ok[(k + 1) % 3] = True
    if sum(ok) == 2:
        k = ok.index(False)
        unit[k] = cross(unit[(k + 1) % 3], unit[(k + 2) % 3])
    r = [v for u in unit for v in u]
    if r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]) < 0:
        k = s.index(max(s)) if any(x <= eps for x in s) else 0
        s[k] = -s[k]
        r[k * 3:k * 3 + 3] = [-v for v in r[k * 3:k * 3 + 3]]
    return [x if abs(x) > eps else 0.0 for x in s], r


def trs(m: list, fallback: list | None = None) -> tuple[list, list, list]:
    """A matrix as glTF's translation, rotation and scale (a mirror as a negative scale)."""
    s, r = scale_rotation(part3(m), fallback)
    return list(m[12:15]), quat_of(r), s


def from_trs(t: list, q: list, s: list) -> list:
    r = rot_of(q)
    return full([r[k] * s[k // 3] for k in range(9)], t)


def slerp(a: list, b: list, u: float) -> list:
    d = sum(x * y for x, y in zip(a, b))
    if d < 0:
        b, d = [-x for x in b], -d
    if d > 0.9995:
        q = [x + (y - x) * u for x, y in zip(a, b)]
    else:
        th = math.acos(d)
        sa, sb = math.sin((1 - u) * th) / math.sin(th), math.sin(u * th) / math.sin(th)
        q = [x * sa + y * sb for x, y in zip(a, b)]
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return [x / n for x in q]


class Buffer:
    def __init__(self):
        self.data = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(self, fmt: str, values: list, kind: str, count: int, ctype: int, target: int | None = None,
            minmax: bool = False) -> int:
        self.data += b"\0" * (-len(self.data) % 4)
        raw = struct.pack(f"<{len(values)}{fmt}", *values)
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target:
            view["target"] = target
        self.data += raw
        self.views.append(view)
        acc = {"bufferView": len(self.views) - 1, "componentType": ctype, "count": count, "type": kind}
        if minmax:
            n = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[kind]
            acc["min"] = [min(values[k::n]) for k in range(n)]
            acc["max"] = [max(values[k::n]) for k in range(n)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def raw(self, data: bytes) -> int:
        """A bufferView of bytes as they are (an image)."""
        self.data += b"\0" * (-len(self.data) % 4)
        self.views.append({"buffer": 0, "byteOffset": len(self.data), "byteLength": len(data)})
        self.data += data
        return len(self.views) - 1


def read_gltf(path: Path) -> tuple[dict, list[bytes]]:
    """A .gltf (data inside, or in files beside it) or a .glb: its JSON and its buffers."""
    data = path.read_bytes()
    bin_chunk = None
    if data[:4] == b"glTF":
        length = struct.unpack_from("<I", data, 8)[0]
        off, gltf = 12, None
        while off + 8 <= min(length, len(data)):
            size, kind = struct.unpack_from("<II", data, off)
            chunk = data[off + 8:off + 8 + size]
            if kind == 0x4E4F534A:
                gltf = json.loads(chunk.decode("utf-8"))
            elif kind == 0x004E4942:
                bin_chunk = chunk
            off += 8 + size + (-size % 4)
        if gltf is None:
            raise GltfError("a .glb without its JSON")
    else:
        gltf = json.loads(data.decode("utf-8-sig"))
    buffers = []
    for b in gltf.get("buffers", []):
        uri = b.get("uri")
        if uri is None:
            if bin_chunk is None:
                raise GltfError("a buffer without data")
            buffers.append(bin_chunk)
        elif uri.startswith("data:"):
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            from urllib.parse import unquote
            buffers.append((path.parent / unquote(uri)).read_bytes())
    return gltf, buffers


COMPONENTS = {5120: ("b", 127.0), 5121: ("B", 255.0), 5122: ("h", 32767.0), 5123: ("H", 65535.0),
              5125: ("I", None), 5126: ("f", None)}


WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def accessor(gltf: dict, buffers: list[bytes], index: int) -> list[list[float]]:
    acc = gltf["accessors"][index]
    n, width = acc["count"], WIDTH[acc["type"]]
    fmt, norm = COMPONENTS[acc["componentType"]]
    if "bufferView" not in acc:
        return [[0.0] * width for _ in range(n)]
    view = gltf["bufferViews"][acc["bufferView"]]
    data = buffers[view["buffer"]]
    size = struct.calcsize(fmt)
    stride = view.get("byteStride") or size * width
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    out = []
    for i in range(n):
        v = list(struct.unpack_from(f"<{width}{fmt}", data, base + i * stride))
        if norm and acc.get("normalized", fmt != "I"):
            v = [max(x / norm, -1.0) for x in v]
        out.append(v)
    return out


class Track:
    """A glTF animation sampler, sampled at a time as the spec says (STEP, LINEAR with slerp, CUBICSPLINE)."""

    def __init__(self, gltf: dict, buffers: list[bytes], sampler: dict, rotation: bool):
        self.times = [t[0] for t in accessor(gltf, buffers, sampler["input"])]
        values = accessor(gltf, buffers, sampler["output"])
        self.mode = sampler.get("interpolation", "LINEAR")
        self.rotation = rotation
        if self.mode == "CUBICSPLINE":
            self.ins, self.values, self.outs = values[0::3], values[1::3], values[2::3]
        else:
            self.values = values[:len(self.times)]

    def __call__(self, t: float) -> list:
        times, vals = self.times, self.values
        if not times:
            return []
        if t <= times[0]:
            return vals[0]
        if t >= times[-1]:
            return vals[len(times) - 1]
        i = bisect.bisect_right(times, t) - 1
        t0, t1 = times[i], times[i + 1]
        u = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        if self.mode == "STEP":
            return vals[i]
        if self.mode == "CUBICSPLINE":
            dt = t1 - t0
            u2, u3 = u * u, u * u * u
            h = (2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u, -2 * u3 + 3 * u2, u3 - u2)
            v = [h[0] * a + h[1] * dt * b + h[2] * c + h[3] * dt * d
                 for a, b, c, d in zip(vals[i], self.outs[i], vals[i + 1], self.ins[i + 1])]
            if self.rotation:
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                v = [x / n for x in v]
            return v
        if self.rotation:
            return slerp(vals[i], vals[i + 1], u)
        return [a + (b - a) * u for a, b in zip(vals[i], vals[i + 1])]
