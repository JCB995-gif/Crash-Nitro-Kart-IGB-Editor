#!/usr/bin/env python3
"""A model out of an Alchemy .igb (Crash Nitro Kart's): its skeleton (bones, parents, bind poses), its meshes
(skinned to the skeleton or rigid, with positions, normals, colours, texcoords, bone weights, triangles), their
textures and the skeleton's animations; and a model's meshes and textures written back into the .igb they came from.

Space: the file's own (Z up, a point times a matrix, matrices as 16 floats row by row). A bone's bind pose is the
inverse of its inverse bind matrix (igSkeleton's, by its blend matrix index); a bone without one (an IK helper) sits
at its translation in its parent's bind. A skin's vertices are in the model's space: a vertex weighted to bones goes
by each one's inverse bind times its pose.

Geometry is igGeometryAttr1_5 over an igVertexArray1_1: streams by slot (0 positions, 1 normals, 2 colours, 11
texcoords) and, for a skin, its weights and its indices into the igBlendMatrixSelect's palette of blend matrices; no
index list: triangle strips one after another through the vertices (their lengths in igPrimLengthArray1_1), every
other triangle turned, or a plain list of triangles. Written back, a mesh's triangles go into strips again.

    igb_model.py <file.igb>                     what the model has
    igb_model.py <file.igb> --obj <out.obj>     its meshes as OBJ (to look at)
"""
from __future__ import annotations

from i18n import tr
import bisect
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import igb_file  # noqa: E402
import igb_texture  # noqa: E402

IDENTITY = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
VF_POS, VF_NRM, VF_COL = 1, 2, 4
S_POS, S_NRM, S_COL, S_UV = 0, 1, 2, 11
PRIM_TRIS, PRIM_STRIPS = 3, 4
SHADOW = 99  # the LOD of a shadow volume's geometry
MAX_STRIP = 64  # the longest strip written (the game's own are shorter)
NEUTRAL = (128, 128, 128, 128)  # a PS2 colour of 1.0

# Fields by their place in the class (igb_file.IGB.get).
G_NAME, G_BOX, G_KIDS, G_ATTRS = 0, 1, 3, 4             # igGroup, igGeometry, igAttrSet, igTransform...
T_MATRIX = 4                                            # igTransform
GA_VA, GA_PRIM, GA_COUNT, GA_OFFSET, GA_LENGTHS = 1, 3, 4, 5, 10   # igGeometryAttr1_5
GA_INDICES = 2                                          # igIndexArray (Xbox)
VA_STREAMS, VA_COUNT, VA_FORMAT, VA_WEIGHTS, VA_INDICES = 0, 1, 4, 5, 6  # igVertexArray1_1
PL_MEM, PL_COUNT = 0, 1                                 # igPrimLengthArray1_1
TB_TEXTURE = 1                                          # igTextureBindAttr
TA_IMAGE = 8                                            # igTextureAttr
IM_W, IM_H, IM_FMT, IM_SIZE, IM_PX, IM_CLUT, IM_BITS, IM_PITCH, IM_NAME = 0, 1, 9, 10, 11, 15, 16, 17, 20  # igImage
CL_DATA = 3                                             # igClut
SI_ROOT = 2                                             # igSceneInfo
BOX_MIN, BOX_MAX = 0, 1                                 # igAABox
SQ_XLATE, SQ_QUAT, SQ_SCALE, SQ_TIMES = 0, 1, 2, 8       # igTransformSequence1_5
SK_NAME, SK_TRANSLATIONS, SK_BONES, SK_INVERSE, SK_COUNT = 0, 1, 2, 3, 4
BI_NAME, BI_PARENT, BI_MATRIX = 0, 1, 2                 # igSkeletonBoneInfo
DB_SKELETONS, DB_ANIMATIONS = 2, 3                     # igAnimationDatabase
AN_NAME, AN_BINDINGS, AN_TRACKS, AN_LENGTH = 0, 2, 3, 7
AB_SKELETON, AB_TRACK_MAP = 0, 1                       # igAnimationBinding
AT_SOURCE, AT_ROTATION = 1, 2                          # igAnimationTrack
BMS_PALETTE, MVM_BONE = 6, 4
SKIN_NAME, SKIN_ROOT = 0, 1


class ModelError(ValueError):
    pass


# Matrices (16 floats row by row, a point times the matrix).

def mul(a: list, b: list) -> list:
    return [a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c] + a[r * 4 + 3] * b[12 + c]
            for r in range(4) for c in range(4)]


def inverse(m: list) -> list:
    """An affine matrix's inverse (the 3x3 by cofactors, then the translation)."""
    a, b, c, d, e, f, g, h, i = m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-20:
        return list(IDENTITY)
    r = [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
         (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
         (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]
    t = m[12:15]
    p = [-(t[0] * r[k] + t[1] * r[3 + k] + t[2] * r[6 + k]) for k in range(3)]
    return [r[0], r[1], r[2], 0.0, r[3], r[4], r[5], 0.0, r[6], r[7], r[8], 0.0, p[0], p[1], p[2], 1.0]


def plain(m: list) -> list:
    """IDENTITY itself for a matrix that is the identity (what is placed by it stays as it is, bit for bit)."""
    return IDENTITY if m == IDENTITY else m


def point(p, m: list) -> tuple:
    x, y, z = p[0], p[1], p[2]
    return (x * m[0] + y * m[4] + z * m[8] + m[12], x * m[1] + y * m[5] + z * m[9] + m[13],
            x * m[2] + y * m[6] + z * m[10] + m[14])


def direction(p, m: list) -> tuple:
    x, y, z = p[0], p[1], p[2]
    return (x * m[0] + y * m[4] + z * m[8], x * m[1] + y * m[5] + z * m[9], x * m[2] + y * m[6] + z * m[10])


def normalised(v) -> tuple:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


def quat_matrix(q) -> list:
    """An Alchemy quaternion (x, y, z, w) as a 3x3 for row vectors: Alchemy's turn the other way round from the
    usual (their conjugate's matrix)."""
    x, y, z, w = -q[0], -q[1], -q[2], q[3]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
            [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
            [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]]


def slerp(a, b, u: float) -> list:
    d = sum(x * y for x, y in zip(a, b))
    if d < 0:
        b, d = [-x for x in b], -d
    if d > 0.9995:
        q = [x + (y - x) * u for x, y in zip(a, b)]
    else:
        th = math.acos(min(1.0, d))
        s0, s1 = math.sin((1 - u) * th) / math.sin(th), math.sin(u * th) / math.sin(th)
        q = [x * s0 + y * s1 for x, y in zip(a, b)]
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return [x / n for x in q]


def facial_bones(bones: list) -> list[int]:
    """Offer actual rig controls, including the naming variants used by CNK's characters."""
    words = ("jaw", "mouth", "eye", "brow", "lip", "cheek", "tongue", "nose", "face", "snout")
    return [i for i, b in enumerate(bones) if any(w in b.name.lower() for w in words)
            or b.name.lower().startswith("ear") or "_ear" in b.name.lower()]


def offset_pose(bones: list, pose: list, offsets: dict[int, tuple]) -> list:
    """Preview local XYZ rotations in degrees and translations on top of an animated pose.

    Rebuild descendants from their original local matrices so moving a jaw also moves
    its teeth; the input pose, rig and cached animation samples remain untouched.
    """
    if not offsets:
        return pose
    out = []
    for i, bone in enumerate(bones):
        parent = bone.parent
        local = mul(pose[i], inverse(pose[parent])) if 0 <= parent < i else list(pose[i])
        if i in offsets:
            rx, ry, rz, tx, ty, tz = offsets[i]
            rotation = list(IDENTITY)
            for axis, angle in enumerate((rx, ry, rz)):
                a = math.radians(angle) / 2
                q = [0.0, 0.0, 0.0, math.cos(a)]
                q[axis] = -math.sin(a)
                r = quat_matrix(q)
                rotation = mul(rotation, [*r[0], 0, *r[1], 0, *r[2], 0, 0, 0, 0, 1])
            local = mul(rotation, local)
            local[12:15] = [local[12] + tx, local[13] + ty, local[14] + tz]
        out.append(mul(local, out[parent]) if 0 <= parent < i else local)
    return out


# The model.

class Bone:
    def __init__(self, name: str, parent: int, bind: list, bm: int = -1, trans=(0.0, 0.0, 0.0)):
        self.name, self.parent, self.bind, self.bm, self.trans = name, parent, bind, bm, tuple(trans)


class Texture:
    def __init__(self, name: str, w: int, h: int, rgba: bytes, image: int = -1):
        self.name, self.w, self.h, self.rgba, self.image = name, w, h, rgba, image
        self.changed = False


class Mesh:
    """A mesh: per vertex a position, a normal, a colour (0-255, 128 the PS2's 1.0), texcoords, its bones' weights
    [(bone, weight)]; triangles; its texture; for a rigid mesh the matrix that places it (or the bone it moves
    with); blend shapes [(name, [delta per vertex])]."""

    def __init__(self, name: str):
        self.name = name
        self.pos: list[tuple] = []
        self.nrm: list[tuple] | None = None
        self.col: list[tuple] | None = None
        self.uv: list[tuple] | None = None
        self.weights: list[list[tuple[int, float]]] | None = None  # None: rigid
        self.tris: list[tuple[int, int, int]] = []
        self.strips: list[int] | None = None  # the vertices' strips as read (they run through all of them)
        self.indices: list[int] | None = None  # Xbox: strips address a separate vertex buffer
        self.color_scale = 2.0  # PS2's 128 is white; Xbox's 255 is white
        self.unset_vertices: set[int] = set()  # runtime-generated geometry; original bytes stay in model.igb
        self.node_path: tuple[int, ...] = ()
        self.texture = -1
        self.matrix = list(IDENTITY)  # a rigid mesh's place (its vertices are written in its own space)
        self.bone = -1                # a rigid mesh moving with a bone (igModelViewMatrixBoneSelect)
        self.lod = 0                  # 0 the most detailed, more the farther ones (igLod's children)
        self.shapes: list[tuple[str, list[tuple]]] = []
        self.attr = -1                # the igGeometryAttr1_5 it came from
        self.skin_name = ""

    @property
    def skinned(self) -> bool:
        return self.weights is not None


class Clip:
    """An animation: its length in seconds and every bone's matrix in the model at a time."""

    def __init__(self, name: str, length: float):
        self.name, self.length = name, length

    def pose(self, t: float) -> list[list]:
        raise NotImplementedError


class Model:
    def __init__(self):
        self.bones: list[Bone] = []
        self.meshes: list[Mesh] = []
        self.textures: list[Texture] = []
        self.platform: str | None = None
        self.clips: list[Clip] = []
        self.notes: list[str] = []
        self.igb: igb_file.IGB | None = None
        self.source = ""
        self.scene_motion = None

    def mesh_matrix(self, mesh: Mesh, clip: Clip | None = None, t: float = 0.0) -> list:
        if self.scene_motion is None or not mesh.node_path or clip is None:
            return mesh.matrix
        return self.scene_motion.world(mesh.node_path, getattr(clip, "scene_phase", 0), t)

    def skin_matrices(self, pose: list[list] | None) -> list[list]:
        """Each bone's inverse bind times its pose (identity at rest)."""
        if pose is None:
            return [list(IDENTITY) for _ in self.bones]
        return [mul(inverse(b.bind), pose[i]) if i < len(pose) else list(IDENTITY) for i, b in enumerate(self.bones)]

    def bounds(self) -> tuple[tuple, tuple]:
        pts = [point(p, m.matrix if not m.skinned and m.bone < 0 else
                     (self.bones[m.bone].bind if 0 <= m.bone < len(self.bones) else IDENTITY))
               for m in self.meshes for p in m.pos[::max(1, len(m.pos) // 400)]]
        pts += [tuple(b.bind[12:15]) for b in self.bones if abs(b.bind[12]) + abs(b.bind[13]) + abs(b.bind[14]) < 1e4]
        if not pts:
            return (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)
        return tuple(min(p[k] for p in pts) for k in range(3)), tuple(max(p[k] for p in pts) for k in range(3))


# Reading.

def node_kids(f: igb_file.IGB, o) -> list:
    """A node's children: the items of every node list it refers to (vvInstancedNode and the light sets have two)."""
    out = []
    for s in sorted(o.slots):
        if o.kinds[s] != b"ObjectRef":
            continue
        t = f.ref(o, s)
        if isinstance(t, igb_file.Obj) and t.is_a("igNodeList"):
            out += [k for k in f.items(t) if isinstance(k, igb_file.Obj)]
    return out


def node_attrs(f: igb_file.IGB, o) -> list:
    lst = f.gref(o, G_ATTRS)
    return [a for a in f.items(lst) if a is not None] if isinstance(lst, igb_file.Obj) and \
        lst.is_a("igAttrList") else []


def picture(f: igb_file.IGB, img) -> tuple[int, int, bytes]:
    """An igImage as RGBA: logical Alchemy palettes (PS2 alpha 0x80 opaque),
    RGB(A), or Xbox DXT. GS CLUT permutations belong to the runtime upload, not this file."""
    w, h = f.get(img, IM_W, 0) or 0, f.get(img, IM_H, 0) or 0
    px = f.gref(img, IM_PX)
    px = px.data if isinstance(px, igb_file.Mem) else b""
    clut = f.gref(img, IM_CLUT)
    bits = f.get(img, IM_BITS, 0)
    fmt = f.get(img, IM_FMT, 0)
    if not (0 < w <= 4096 and 0 < h <= 4096):
        return 8, 8, bytes((128, 128, 128, 255)) * 64
    if fmt in (14, 15, 16):
        return w, h, igb_texture.decode(w, h, px, fmt)
    out = bytearray()
    if isinstance(clut, igb_file.Obj) and isinstance(f.gref(clut, CL_DATA), igb_file.Mem) and bits in (4, 8):
        raw = f.gref(clut, CL_DATA).data
        alpha_scale = 2 if max(raw[3::4], default=0) <= 128 else 1
        pal = [raw[4 * i:4 * i + 4] for i in range(len(raw) // 4)]
        pal += [b"\0\0\0\x80"] * (256 - len(pal))
        need = w * h if bits == 8 else (w * h + 1) // 2
        if len(px) >= need:
            for i in range(w * h):
                k = px[i] if bits == 8 else (px[i >> 1] >> (4 * (i & 1))) & 15
                r, g, b, a = pal[k]
                out += bytes((r, g, b, min(255, a * alpha_scale)))
            return w, h, bytes(out)
    if fmt == 7 and len(px) >= 4 * w * h:
        top = max(px[3:4 * w * h:4], default=0)
        scale = 2 if top <= 0x80 else 1
        for i in range(w * h):
            r, g, b, a = px[4 * i:4 * i + 4]
            out += bytes((r, g, b, min(255, a * scale)))
        return w, h, bytes(out)
    if fmt == 5 and len(px) >= 3 * w * h:
        for i in range(w * h):
            out += px[3 * i:3 * i + 3] + b"\xff"
        return w, h, bytes(out)
    if fmt == 0 and len(px) >= w * h:  # L8
        return w, h, b"".join(bytes((v, v, v, 255)) for v in px[:w * h])
    if fmt == 8 and len(px) >= 2 * w * h:  # RGBA5551, R at the high end, A in bit 0
        for (value,) in struct.iter_unpack(f.e + "H", px[:2 * w * h]):
            rgb = tuple((((value >> shift) & 31) * 255 + 15) // 31 for shift in (11, 6, 1))
            out += bytes((*rgb, 255 if value & 1 else 0))
        return w, h, bytes(out)
    raise ModelError(tr('image #{0}: unsupported format {1} or truncated data ({2}×{3})', img.index, fmt, w, h))


def read_skeleton(f: igb_file.IGB, sk) -> list[Bone]:
    infos = [b for b in f.items(f.gref(sk, SK_BONES)) if b is not None]
    tm = f.gref(sk, SK_TRANSLATIONS)
    tm = tm.data if isinstance(tm, igb_file.Mem) else b""
    inv = f.gref(sk, SK_INVERSE)
    inv = inv.data if isinstance(inv, igb_file.Mem) else b""
    bones = []
    for k, b in enumerate(infos):
        t = struct.unpack_from(f.e + "3f", tm, 12 * k) if 12 * k + 12 <= len(tm) else (0.0, 0.0, 0.0)
        if not all(abs(x) < 1e6 for x in t):  # unset: 0xCDCDCDCD
            t = (0.0, 0.0, 0.0)
        bm = f.get(b, BI_MATRIX, -1)
        parent = f.get(b, BI_PARENT, -1)
        parent = parent if 0 <= parent < k else -1
        if 0 <= bm < len(inv) // 64:
            bind = inverse(list(struct.unpack_from(f.e + "16f", inv, 64 * bm)))
        else:  # no bind of its own: its translation in its parent's
            pm = bones[parent].bind if parent >= 0 else IDENTITY
            bind = list(pm[:12]) + list(point(t, pm)) + [1.0]
        bones.append(Bone(str(f.get(b, BI_NAME, "")) or f"bone{k}", parent, bind, bm, t))
    return bones


def main_skeleton(f: igb_file.IGB):
    db = f.of("igAnimationDatabase")
    for d in db:
        sks = [s for s in f.items(f.gref(d, DB_SKELETONS)) if s is not None]
        if sks:
            return sks[0]
    sks = f.of("igSkeleton")
    return max(sks, key=lambda s: f.get(s, SK_COUNT, 0)) if sks else None


class _Walk:
    """The geometries under a node, each with what reaches it from above: its texture, its skin's palette, its
    place, the bone it moves with, its LOD."""

    def __init__(self, f: igb_file.IGB):
        self.f = f
        self.out: list[tuple] = []
        self.seen: set[int] = set()

    def walk(self, o, tex, palette, matrix, bone, lod, skin, depth=0, path=()):
        f = self.f
        if o is None or depth > 64 or o.index in path:
            return
        path = path + (o.index,)
        if o.is_a("igTransform"):
            m = f.get(o, T_MATRIX)
            if isinstance(m, tuple) and len(m) == 16:
                matrix = mul(list(m), matrix)
        if o.is_a("igBlendMatrixSelect"):
            il = f.gref(o, BMS_PALETTE)
            palette = [v[0] for v in f.data_list(il, "i")]
            skin = skin or str(f.get(o, G_NAME, "") or "")
        if o.is_a("igModelViewMatrixBoneSelect"):
            bone = f.get(o, MVM_BONE, -1)
        for a in node_attrs(f, o):
            if a.is_a("igTextureBindAttr"):
                ta = f.gref(a, TB_TEXTURE)
                img = f.gref(ta, TA_IMAGE) if isinstance(ta, igb_file.Obj) else None
                if isinstance(img, igb_file.Obj):
                    tex = img
        if o.is_a("igGeometry"):
            if o.index not in self.seen:
                self.seen.add(o.index)
                transforms = tuple(i for i in path if f.objects[i].is_a("igTransform"))
                self.out.append((o, tex, palette, matrix, bone, lod, skin, transforms))
            return
        kids = node_kids(f, o)
        is_lod = o.is_a("igLod")
        for k, kid in enumerate(kids):
            self.walk(kid, tex, palette, matrix, bone, max(lod, k) if is_lod and lod < SHADOW else lod, skin,
                      depth + 1, path)


def strip_triangles(n_lengths: list[int], prim: int, count: int, nverts: int) -> list[tuple[int, int, int]]:
    """The triangles of strips one after another through the vertices (every other one turned), or of a list."""
    tris = []
    if prim == PRIM_STRIPS:
        at = 0
        for L in n_lengths:
            for k in range(L - 2):
                a, b, c = at + k, at + k + 1, at + k + 2
                if k % 2:
                    b, c = c, b
                if max(a, b, c) < nverts:
                    tris.append((a, b, c))
            at += L
    elif prim == PRIM_TRIS:
        for k in range(min(count, nverts // 3)):
            tris.append((3 * k, 3 * k + 1, 3 * k + 2))
    return tris


def streams(f: igb_file.IGB, va) -> dict[int, igb_file.Mem]:
    """The vertex array's streams by slot (0 positions, 1 normals, 2 colours, 11 texcoords)."""
    table = f.gref(va, VA_STREAMS)
    out = {}
    if not isinstance(table, igb_file.Mem):
        return out
    for k in range(len(table.data) // 4):
        v = struct.unpack_from(f.e + "i", table.data, 4 * k)[0]
        if 0 <= v < len(f.objects) and isinstance(f.objects[v], igb_file.Mem):
            out[k] = f.objects[v]
    return out


def read_geometry(f: igb_file.IGB, attr, palette: list[int] | None, bm_bone: dict[int, int]) -> Mesh:
    va = f.gref(attr, GA_VA)
    m = Mesh("")
    m.attr = attr.index
    if not isinstance(va, igb_file.Obj):
        return m
    n, fmt = f.get(va, VA_COUNT, 0), f.get(va, VA_FORMAT, 0)
    st = streams(f, va)
    e = f.e
    if S_POS not in st or len(st[S_POS].data) < 12 * n:
        return m
    m.pos = [struct.unpack_from(e + "3f", st[S_POS].data, 12 * i) for i in range(n)]
    if S_NRM in st and len(st[S_NRM].data) >= 12 * n:
        m.nrm = [struct.unpack_from(e + "3f", st[S_NRM].data, 12 * i) for i in range(n)]
    if S_COL in st and len(st[S_COL].data) >= 4 * n:
        m.col = [tuple(st[S_COL].data[4 * i:4 * i + 4]) for i in range(n)]
    if S_UV in st and len(st[S_UV].data) >= 8 * n:
        m.uv = [struct.unpack_from(e + "2f", st[S_UV].data, 8 * i) for i in range(n)]
    nw, ni = (fmt >> 4) & 15, (fmt >> 8) & 15
    bw, bi = f.gref(va, VA_WEIGHTS), f.gref(va, VA_INDICES)
    if palette is not None and ni and isinstance(bi, igb_file.Mem):
        m.weights = []
        for i in range(n):
            ws = struct.unpack_from(e + f"{nw}f", bw.data, 4 * nw * i) if isinstance(bw, igb_file.Mem) and nw and \
                4 * nw * (i + 1) <= len(bw.data) else ()
            ix = tuple(bi.data[ni * i:ni * i + ni])
            if not nw:
                ws = (1.0,)
            if len(ws) == ni - 1:  # the last weight is what the others leave
                ws = ws + (1.0 - sum(ws),)
            vw = []
            for j, w in zip(ix, ws):  # (a weight of 0 kept: its index goes back as it was)
                if w >= 0:
                    bm = palette[j] if j < len(palette) else j
                    vw.append((bm_bone.get(bm, 0), w))
            m.weights.append(vw or [(bm_bone.get(palette[0] if palette else 0, 0), 1.0)])
    prim, count = f.get(attr, GA_PRIM, PRIM_STRIPS), f.get(attr, GA_COUNT, 0)
    index_array = f.gref(attr, GA_INDICES)
    order = list(range(n))
    if isinstance(index_array, igb_file.Obj):
        mem = f.gref(index_array, 0)
        size = f.get(index_array, 1, 0)
        # CNK's igIndexArray enum 0 is unsigned short, 1 unsigned int.
        code = {0: "H", 1: "I"}.get(f.get(index_array, 2, 0))
        if not isinstance(mem, igb_file.Mem) or code is None:
            raise ModelError(f"{attr!r}: unsupported index array")
        stride = struct.calcsize(code)
        if size * stride > len(mem.data):
            raise ModelError(f"{attr!r}: truncated index array")
        order = list(struct.unpack_from(e + f"{size}{code}", mem.data))
        m.indices = order
        m.color_scale = 1.0
    offset = f.get(attr, GA_OFFSET, 0)
    order = order[offset:]
    if prim == PRIM_STRIPS:
        pl = f.gref(attr, GA_LENGTHS)
        lm = f.gref(pl, PL_MEM) if isinstance(pl, igb_file.Obj) else None
        k = f.get(pl, PL_COUNT, 0) if isinstance(pl, igb_file.Obj) else 0
        lens = list(struct.unpack_from(e + f"{k}I", lm.data, 0)) if isinstance(lm, igb_file.Mem) and \
            4 * k <= len(lm.data) else []
        m.strips = lens
        m.tris = strip_triangles(lens[:count], prim, count, len(order))
    else:
        m.tris = strip_triangles([], prim, count, len(order))
    m.tris = [tuple(order[i] for i in t) for t in m.tris]
    if any(i >= n for t in m.tris for i in t):
        raise ModelError(f"{attr!r}: vertex index outside buffer ({n} vertices)")
    # Some effects ship with NaN placeholders filled by the game at runtime. Keep
    # their raw memory in the IGB inspector, but never feed NaNs to the camera/GPU.
    m.unset_vertices = {i for i, p in enumerate(m.pos) if not all(math.isfinite(v) for v in p)}
    if m.unset_vertices:
        m.tris = [t for t in m.tris if not any(i in m.unset_vertices for i in t)]
        m.pos = [(0.0, 0.0, 0.0) if i in m.unset_vertices else p for i, p in enumerate(m.pos)]
    # Triangles of nothing (the strips' joins) left out.
    m.tris = [t for t in m.tris if len({m.pos[t[0]], m.pos[t[1]], m.pos[t[2]]}) == 3]
    return m


def from_igb(data: bytes, name: str = "", platform: str | None = None) -> Model:
    f = igb_file.IGB(data)
    model = Model()
    model.igb, model.source = f, name
    parts = name.replace("\\", "/").lower().split("/")
    model.platform = platform or next((p for p in parts if p in ("ps2", "xbox")), None)
    sk = main_skeleton(f)
    if sk is not None:
        model.bones = read_skeleton(f, sk)
    bm_bone = {b.bm: i for i, b in enumerate(model.bones) if b.bm >= 0}
    walk = _Walk(f)
    for si in f.of("igSceneInfo"):  # the scene first: what it says above a skin (its texture) holds for it
        root = f.gref(si, SI_ROOT)
        if isinstance(root, igb_file.Obj):
            walk.walk(root, None, None, list(IDENTITY), -1, 0, "")
    for skin in f.of("igSkin"):  # then the skins reached from the animation database only
        root = f.gref(skin, SKIN_ROOT)
        if isinstance(root, igb_file.Obj):
            walk.walk(root, None, None, list(IDENTITY), -1, 0, str(f.get(skin, SKIN_NAME, "") or ""))
    for vol in f.of("vvStencilShadowVolume"):  # the karts' shadow volumes (from a shadow attribute, not seen)
        walk.walk(vol, None, None, list(IDENTITY), -1, SHADOW, "")
    for g in f.of("igGeometry"):  # anything left hanging
        if g.index not in walk.seen:
            walk.walk(g, None, None, list(IDENTITY), -1, 0, "")
    default_tex = None
    for si in f.of("igSceneInfo"):
        root = f.gref(si, SI_ROOT)
        for a in node_attrs(f, root) if isinstance(root, igb_file.Obj) else []:
            if a.is_a("igTextureBindAttr") and isinstance(f.gref(f.gref(a, TB_TEXTURE), TA_IMAGE), igb_file.Obj):
                default_tex = f.gref(f.gref(a, TB_TEXTURE), TA_IMAGE)
    images = f.of("igImage")
    tex_index: dict[int, int] = {}

    def texture(img) -> int:
        if img is None:
            img = default_tex if default_tex is not None else (images[0] if len(images) == 1 else None)
        if img is None:
            return -1
        if img.index not in tex_index:
            w, h, rgba = picture(f, img)
            nm = str(f.get(img, IM_NAME, "") or "").replace("\\", "/").rsplit("/", 1)[-1] or f"texture{img.index}"
            tex_index[img.index] = len(model.textures)
            model.textures.append(Texture(nm, w, h, rgba, img.index))
        return tex_index[img.index]

    names: dict[str, int] = {}
    for g, tex, palette, matrix, bone, lod, skin, transforms in walk.out:
        for attr in node_attrs(f, g):
            if not attr.is_a("igGeometryAttr"):
                continue
            m = read_geometry(f, attr, palette, bm_bone)
            if model.platform:
                m.color_scale = 1.0 if model.platform == "xbox" else 2.0
            if not m.pos:
                continue
            base = str(f.get(g, G_NAME, "") or "") or f"geometry{g.index}"
            names[base] = names.get(base, 0) + 1
            m.name = base if names[base] == 1 else f"{base}#{names[base]}"
            m.texture = texture(tex) if lod != SHADOW else -1
            m.matrix, m.lod, m.skin_name = matrix, lod, skin
            m.node_path = transforms
            if m.weights is None and 0 <= bone < len(model.bones):
                m.bone = bone
            model.meshes.append(m)
            if m.unset_vertices:
                model.notes.append(tr('{0}: {1} undefined vertices in the file (generated by the game); original bytes are available in IGB data.', m.name, len(m.unset_vertices)))
    for img in images:  # textures no mesh uses, to see and to change too
        texture(img)
    model.clips = read_clips(f, model.bones)
    from igb_scene import Motion
    motion = Motion(f)
    if motion.channels:
        model.scene_motion = motion
        model.notes.extend(motion.notes)
        if not model.clips:
            model.clips = motion.clips()
        else:
            ordered = []
            for database in f.of("igAnimationDatabase"):
                ordered.extend(a.index for a in f.items(f.gref(database, DB_ANIMATIONS)) if isinstance(a, igb_file.Obj))
            for clip in model.clips:
                clip.scene_phase = ordered.index(clip.object_index) if clip.object_index in ordered else 0
    for m in model.meshes:
        if m.skinned and m.weights and not model.bones:
            m.weights = None
    return model


# Animations (igAnimation: per bone a track of quaternions and translations at times).

class IgbClip(Clip):
    def __init__(self, f: igb_file.IGB, anim, bones: list[Bone]):
        super().__init__(str(f.get(anim, AN_NAME, "") or f"anim{anim.index}"),
                         (f.get(anim, AN_LENGTH, 0) or 0) / 1e9)
        self.bones = bones
        self.object_index = anim.index
        bindings = [b for b in f.items(f.gref(anim, AN_BINDINGS)) if b is not None]
        # Additional bindings may target exporter/alternate rigs. Prefer the
        # skeleton used by the skin instead of whichever binding was serialized first.
        main = main_skeleton(f)
        binding = next((b for b in bindings if f.gref(b, AB_SKELETON) is main), bindings[0] if bindings else None)
        own = read_skeleton(f, f.gref(binding, AB_SKELETON)) if binding is not None and \
            isinstance(f.gref(binding, AB_SKELETON), igb_file.Obj) else bones
        mem = f.gref(binding, AB_TRACK_MAP) if binding is not None else None
        track_of = list(struct.unpack_from(f.e + f"{len(mem.data) // 4}i", mem.data, 0)) \
            if isinstance(mem, igb_file.Mem) else []
        # The clip's skeleton's bones on the model's (by name, when they are not the same list).
        by_name = {b.name: i for i, b in enumerate(own)}
        self.of_bone = []
        for i, b in enumerate(bones):
            k = i if i < len(own) and own[i].name == b.name else by_name.get(b.name, -1)
            self.of_bone.append(track_of[k] if 0 <= k < len(track_of) else -1)
        self.trans = [own[by_name[b.name]].trans if b.name in by_name else b.trans for b in bones]
        self.tracks = []
        for t in f.items(f.gref(anim, AN_TRACKS)):
            if t is None:
                self.tracks.append(None)
                continue
            src = f.gref(t, AT_SOURCE)
            const_q = tuple(f.get(t, AT_ROTATION, (0.0, 0.0, 0.0, 1.0)))
            if not isinstance(src, igb_file.Obj):
                self.tracks.append(([], [const_q], [], []))
                continue
            times = [v[0] / 1e9 for v in f.data_list(f.gref(src, SQ_TIMES), "q")]
            quats = [tuple(v) for v in f.data_list(f.gref(src, SQ_QUAT), "4f")]
            xl = [tuple(v) for v in f.data_list(f.gref(src, SQ_XLATE), "3f")]
            scales = [tuple(v) for v in f.data_list(f.gref(src, SQ_SCALE), "3f")]
            self.tracks.append((times, quats or [const_q], xl, scales))
        if self.length <= 0:
            self.length = max((tr[0][-1] for tr in self.tracks if tr and tr[0]), default=1.0)
        self._cache: dict[int, list] = {}

    @staticmethod
    def _at(times: list, values: list, t: float):
        if not values:
            return None
        if len(values) == 1 or not times or t <= times[0]:
            return values[0]
        i = bisect.bisect_right(times, t) - 1
        if i >= min(len(times), len(values)) - 1:
            return values[min(len(times), len(values)) - 1]
        u = (t - times[i]) / ((times[i + 1] - times[i]) or 1.0)
        a, b = values[i], values[i + 1]
        return slerp(a, b, u) if len(a) == 4 else [x + (y - x) * u for x, y in zip(a, b)]

    def pose(self, t: float) -> list[list]:
        key = round(t * 1000)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = []
        for b, bone in enumerate(self.bones):
            k = self.of_bone[b]
            q, tr, scale = None, None, None
            if 0 <= k < len(self.tracks) and self.tracks[k]:
                times, quats, xl, scales = self.tracks[k]
                q = self._at(times, quats, t)
                tr = self._at(times, xl, t) if xl else None
                scale = self._at(times, scales, t) if scales else None
            if q is None:
                if bone.parent >= 0:  # no track: the bind's turn from its parent
                    loc = mul(bone.bind, inverse(self.bones[bone.parent].bind))
                    r = [loc[0:3], loc[4:7], loc[8:11]]
                else:
                    r = [bone.bind[0:3], bone.bind[4:7], bone.bind[8:11]]
            else:
                r = quat_matrix(q)
            if scale is not None:
                r = [[v * scale[axis] for v in row] for axis, row in enumerate(r)]
            tr = tr or self.trans[b]
            m = [*r[0], 0.0, *r[1], 0.0, *r[2], 0.0, *tr, 1.0]
            p = bone.parent
            out.append(mul(m, out[p]) if 0 <= p < len(out) else m)
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[key] = out
        return out


def read_clips(f: igb_file.IGB, bones: list[Bone]) -> list[Clip]:
    if not bones:
        return []
    out = []
    for a in f.of("igAnimation"):
        # Never silently hide an animation that failed to decode.
        out.append(IgbClip(f, a, bones))
    return out


# Writing a model's meshes back into its .igb.

def stripify(tris: list[tuple[int, int, int]], longest: int = MAX_STRIP) -> list[list[int]]:
    """Triangles as strips (each a list of vertex ids, the decoder turning every other triangle), greedy: each strip
    starts at the triangle with the fewest neighbours left (the mesh's edges are used up from outside in), goes
    the longest of its three ways round, and grows at both ends (the end before its first triangle walked as a
    strip of its own the other way, then turned round; a strip with an odd number of triangles before it gets its
    first vertex twice, a triangle of nothing that keeps the others facing the right way)."""
    import heapq
    tris = [t for t in tris if len(set(t)) == 3]
    by_edge: dict[tuple[int, int], list[int]] = {}
    for k, (a, b, c) in enumerate(tris):
        for u, v in ((a, b), (b, c), (c, a)):
            by_edge.setdefault((min(u, v), max(u, v)), []).append(k)
    near = [set() for _ in tris]
    for ts in by_edge.values():
        for t in ts:
            near[t].update(x for x in ts if x != t)
    used = [False] * len(tris)
    free = [len(n) for n in near]
    heap = [(free[k], k) for k in range(len(tris))]
    heapq.heapify(heap)

    def same_way(t, s) -> bool:
        return t in (s, (s[1], s[2], s[0]), (s[2], s[0], s[1]))

    def grow(strip: list[int], taken: set, flip: bool, room: int) -> None:
        """More triangles across the strip's last edge (flip: the triangles faced the other way round)."""
        while room > 0:
            u, v = strip[-2], strip[-1]
            k = len(strip) - 2  # the next triangle's place in the strip
            nxt = None
            for t in by_edge.get((min(u, v), max(u, v)), ()):
                if used[t] or t in taken:
                    continue
                x = next(i for i in tris[t] if i != u and i != v)
                want = (u, v, x) if (k % 2 == 0) != flip else (u, x, v)
                if same_way(want, tris[t]):
                    nxt = (t, x)
                    break
            if nxt is None:
                return
            taken.add(nxt[0])
            strip.append(nxt[1])
            room -= 1

    strips = []
    while heap:
        _, start = heapq.heappop(heap)
        if used[start]:
            continue
        best = None
        a, b, c = tris[start]
        for s0 in ((a, b, c), (b, c, a), (c, a, b)):
            fwd, taken = list(s0), {start}
            grow(fwd, taken, False, longest - 3)
            # Before the first triangle: from it the other way round (c b a ...), walked, then turned round.
            back = [s0[2], s0[1], s0[0]]
            grow(back, taken, True, longest - len(fwd))
            pre = back[3:][::-1]  # the vertices before a b c
            strip = pre + fwd
            if len(pre) % 2:  # an odd number of triangles before: one of nothing first, to keep the faces right
                strip = [strip[0]] + strip
            if best is None or len(taken) > len(best[1]) or \
                    (len(taken) == len(best[1]) and len(strip) < len(best[0])):
                best = (strip, taken)
        for t in best[1]:
            used[t] = True
            for n in near[t]:
                if not used[n]:
                    free[n] -= 1
                    heapq.heappush(heap, (free[n], n))
        strips.append(best[0])
    return strips


def weld(m: Mesh) -> list[tuple[int, int, int]]:
    """The mesh's triangles over one vertex for each set of values (a strip runs on across the vertices that are
    the same in all but their number: copies made by the strips, or by Blender's import)."""
    def r(v, d):
        return tuple(round(x, d) + 0.0 for x in v)
    first: dict[tuple, int] = {}
    same = []
    for i, p in enumerate(m.pos):  # (values that differ by a float's rounding are the same: Blender's normals)
        key = (r(p, 5), r(m.nrm[i], 3) if m.nrm else None, r(m.uv[i], 5) if m.uv else None,
               m.col[i] if m.col else None,
               tuple(sorted((b, round(w, 4)) for b, w in m.weights[i] if w > 0)) if m.weights is not None else None)
        same.append(first.setdefault(key, i))
    return [(same[a], same[b], same[c]) for a, b, c in m.tris]


def _pack(e: str, fmt: str, rows) -> bytes:
    return b"".join(struct.pack(e + fmt, *r) for r in rows)


def _box(f: igb_file.IGB, box, pts) -> None:
    if isinstance(box, igb_file.Obj) and box.is_a("igAABox") and pts:
        lo = tuple(min(p[k] for p in pts) for k in range(3))
        hi = tuple(max(p[k] for p in pts) for k in range(3))
        f.put(box, BOX_MIN, lo)
        f.put(box, BOX_MAX, hi)


def write_mesh(f: igb_file.IGB, model: Model, old: Mesh, new: Mesh, notes: list[str]) -> None:
    """A mesh read from the file (old) given new's vertices and triangles: the vertex array's streams, the weights
    and palette, the strips. new is in the model's space (old's matrix or bone taken off here)."""
    attr = f.objects[old.attr]
    va = f.gref(attr, GA_VA)
    e = f.e
    fmt = f.get(va, VA_FORMAT, 0)
    st = streams(f, va)
    # The order the vertices go in: the old one when nothing but the vertices' values changed, else new strips.
    prim = f.get(attr, GA_PRIM, PRIM_STRIPS)
    same = (len(new.pos) == len(old.pos) and sorted(new.tris) == sorted(old.tris) and
            (prim != PRIM_STRIPS or old.strips is not None and new.strips == old.strips))
    indexed = isinstance(f.gref(attr, GA_INDICES), igb_file.Obj)
    if indexed:
        same = len(new.pos) == len(old.pos) and sorted(new.tris) == sorted(old.tris)
    if same:
        order, lengths = list(range(len(new.pos))), list(old.strips or [])
    elif prim == PRIM_TRIS:
        order, lengths = [i for triangle in new.tris for i in triangle], []
    else:
        strips = stripify(weld(new))
        order = [i for s in strips for i in s]
        lengths = [len(s) for s in strips]
    if not order:
        notes.append(tr('{0}: no triangles; left unchanged', old.name))
        return
    if len(order) > 65535:
        raise ModelError(tr('{0}: {1} strip vertices (too many)', old.name, len(order)))
    # Back into the mesh's own space.
    if old.skinned:
        place = IDENTITY
    elif 0 <= old.bone < len(model.bones):
        place = plain(inverse(model.bones[old.bone].bind))
    else:
        place = plain(inverse(old.matrix))
    pos = [new.pos[i] if place is IDENTITY else point(new.pos[i], place) for i in order]
    st[S_POS].data = _pack(e, "3f", pos)
    st[S_POS].size = len(st[S_POS].data)
    if S_NRM in st:
        nrm = new.nrm or [(0.0, 0.0, 1.0)] * len(new.pos)
        st[S_NRM].data = _pack(e, "3f", (nrm[i] if place is IDENTITY else normalised(direction(nrm[i], place))
                                         for i in order))
        st[S_NRM].size = len(st[S_NRM].data)
    if S_COL in st:
        neutral = (255,) * 4 if old.color_scale == 1.0 else NEUTRAL
        col = new.col or [neutral] * len(new.pos)
        st[S_COL].data = b"".join(bytes(max(0, min(255, int(c))) for c in col[i]) for i in order)
        st[S_COL].size = len(st[S_COL].data)
    if S_UV in st:
        uv = new.uv or [(0.0, 0.0)] * len(new.pos)
        st[S_UV].data = _pack(e, "2f", (uv[i] for i in order))
        st[S_UV].size = len(st[S_UV].data)
    nw, ni = (fmt >> 4) & 15, (fmt >> 8) & 15
    def weight_totals(row):
        totals = {}
        for bone, weight in row:
            if weight > 0:
                totals[bone] = totals.get(bone, 0.0) + weight
        total = sum(totals.values()) or 1.0
        return {bone: weight / total for bone, weight in totals.items()}
    def equivalent_weights(a, b):
        a, b = weight_totals(a), weight_totals(b)
        return all(math.isclose(a.get(bone, 0.0), b.get(bone, 0.0), rel_tol=1e-6, abs_tol=1e-7)
                   for bone in a.keys() | b.keys())
    same_weights = same and old.weights is not None and new.weights is not None and \
        len(old.weights) == len(new.weights) and all(equivalent_weights(a, b)
                                                   for a, b in zip(old.weights, new.weights))
    if old.skinned and ni and not same_weights:
        bms = [model.bones[i].bm for i in range(len(model.bones))]

        def bm_of(bone: int) -> int:  # a bone without a blend matrix: the nearest parent with one
            while 0 <= bone < len(bms) and bms[bone] < 0:
                bone = model.bones[bone].parent
            return bms[bone] if 0 <= bone < len(bms) else 0

        # The palette: the old one, the blend matrices the new weights need after it.
        bms_node = None
        for o in f.of("igBlendMatrixSelect"):
            if any(isinstance(g, igb_file.Obj) and old.attr in [a.index for a in node_attrs(f, g)]
                   for g in _geometries_under(f, o)):
                bms_node = o
                break
        il = f.gref(bms_node, BMS_PALETTE) if bms_node is not None else None
        palette = [v[0] for v in f.data_list(il, "i")]
        slot = {bm: k for k, bm in enumerate(palette)}
        weights = new.weights or [[(0, 1.0)]] * len(new.pos)
        dropped = 0
        wrows, irows = [], []
        for i in order:
            acc: dict[int, float] = {}
            for b, w in weights[i]:
                if w >= 0:
                    bm = bm_of(b)
                    acc[bm] = acc.get(bm, 0.0) + w
            top = sorted(acc.items(), key=lambda x: -x[1])
            if sum(1 for _, w in top if w > 0) > ni:
                dropped += 1
            top = top[:ni] or [(palette[0] if palette else 0, 1.0)]
            if top[0][1] <= 0:
                top = [(top[0][0], 1.0)]
            top = [(bm if w > 0 or bm in slot else top[0][0], w) for bm, w in top]
            total = sum(w for _, w in top) or 1.0
            if abs(total - 1.0) < 1e-6:  # as they are (back unchanged)
                total = 1.0
            top = [(bm, w / total) for bm, w in top]
            for bm, _ in top:
                if bm not in slot:
                    slot[bm] = len(palette)
                    palette.append(bm)
            top += [(top[0][0], 0.0)] * (ni - len(top))
            irows.append(bytes(slot[bm] for bm, _ in top))
            ws = [w for _, w in top]
            wrows.append(ws[:nw] + [0.0] * (nw - len(ws[:nw])))
        if max(palette, default=0) > 255 or len(palette) > 255:
            raise ModelError(tr('{0}: too many palette matrices ({1})', old.name, len(palette)))
        if dropped:
            notes.append(tr('{0}: {1} vertices had more than {2} bones; kept the {3} strongest influences', old.name, dropped, ni, ni))
        bw, bi = f.gref(va, VA_WEIGHTS), f.gref(va, VA_INDICES)
        if isinstance(bw, igb_file.Mem) and nw:
            bw.data = b"".join(struct.pack(e + f"{nw}f", *r) for r in wrows)
            bw.size = len(bw.data)
        bi.data = b"".join(irows)
        bi.size = len(bi.data)
        if il is not None and len(palette) != len(f.data_list(il, "i")):
            f.set_list(il, struct.pack(e + f"{len(palette)}i", *palette), len(palette))
            notes.append(tr('{0}: the bone palette now contains {1} matrices', old.name, len(palette)))
    f.put(va, VA_COUNT, len(order))
    notes.append(tr('{0}: {1} triangles, {2} vertices in {3} strips', old.name, len(new.tris), len(order), len(lengths))
                 + (tr(' (unchanged)') if same else ""))
    if prim == PRIM_STRIPS and not (indexed and same):
        pl = f.gref(attr, GA_LENGTHS)
        lm = f.gref(pl, PL_MEM)
        lm.data = struct.pack(e + f"{len(lengths)}I", *lengths)
        lm.size = len(lm.data)
        f.put(pl, PL_COUNT, len(lengths))
        f.put(attr, GA_COUNT, len(lengths))
    elif prim == PRIM_TRIS and not same:
        f.put(attr, GA_COUNT, len(order) // 3)
    if indexed and not same:
        ia = f.gref(attr, GA_INDICES)
        mem = f.gref(ia, 0)
        code = {0: "H", 1: "I"}[f.get(ia, 2, 0)]
        size = f.get(va, VA_COUNT)
        if code == "H" and size > 65535:
            raise ModelError(tr('{0}: too many vertices for 16-bit indices', old.name))
        mem.data = struct.pack(e + f"{size}{code}", *range(size))
        mem.size = len(mem.data)
        f.put(ia, 1, size)
        f.put(attr, GA_OFFSET, 0)
    for g in f.of("igGeometry"):
        if old.attr in [a.index for a in node_attrs(f, g)]:
            _box(f, f.gref(g, G_BOX), pos)


def _geometries_under(f: igb_file.IGB, o, depth: int = 0) -> list:
    if o is None or depth > 64:
        return []
    if o.is_a("igGeometry"):
        return [o]
    return [g for k in node_kids(f, o) for g in _geometries_under(f, k, depth + 1)]


def encode_picture(f: igb_file.IGB, img, w: int, h: int, rgba: bytes) -> None:
    """A picture into an igImage of its own size and kind (paletted: a new palette for it)."""
    import png_util  # noqa: E402 - the palette's median cut
    iw, ih = f.get(img, IM_W, 0), f.get(img, IM_H, 0)
    if (w, h) != (iw, ih):  # to the image's size (nearest texel)
        rgba = b"".join(rgba[4 * ((y * h // ih) * w + (x * w // iw)):4 * ((y * h // ih) * w + (x * w // iw)) + 4]
                        for y in range(ih) for x in range(iw))
    px = f.gref(img, IM_PX)
    clut = f.gref(img, IM_CLUT)
    bits = f.get(img, IM_BITS, 0)
    fmt = f.get(img, IM_FMT, 0)
    if (fmt in (0, 5, 7, 8, 14, 15, 16) or isinstance(clut, igb_file.Obj) and bits in (4, 8)) \
            and picture(f, img)[2] == rgba:
        return
    pixels = [tuple(rgba[4 * i:4 * i + 4]) for i in range(iw * ih)]
    if fmt in (14, 15, 16):
        encoded = igb_texture.encode(iw, ih, rgba, fmt)
        px.data = encoded + px.data[len(encoded):]
    elif isinstance(clut, igb_file.Obj) and bits in (4, 8):
        cd = f.gref(clut, CL_DATA)
        n = len(cd.data) // 4
        palette, index = png_util.quantise(pixels, min(n, 1 << bits))
        palette = list(palette) + [(0, 0, 0, 255)] * (n - len(palette))
        alpha_scale = 2 if max(cd.data[3::4], default=0) <= 128 else 1
        entries = [bytes((r, g, b, (a + alpha_scale // 2) // alpha_scale)) for r, g, b, a in palette]
        cd.data = b"".join(entries)
        if bits == 8:
            px.data = bytes(index) + px.data[len(index):]
        else:
            out = bytearray(px.data)
            for i in range(0, len(index), 2):
                out[i >> 1] = index[i] | ((index[i + 1] if i + 1 < len(index) else 0) << 4)
            px.data = bytes(out)
    elif fmt == 7:
        alpha_scale = 2 if max(px.data[3:4 * iw * ih:4], default=0) <= 128 else 1
        px.data = b"".join(bytes((r, g, b, (a + alpha_scale // 2) // alpha_scale)) for r, g, b, a in pixels) + px.data[4 * iw * ih:]
    elif fmt == 5:
        px.data = b"".join(bytes((r, g, b)) for r, g, b, a in pixels) + px.data[3 * iw * ih:]
    elif fmt == 0:
        px.data = bytes((77 * r + 150 * g + 29 * b + 128) >> 8 for r, g, b, a in pixels) + px.data[iw * ih:]
    elif fmt == 8:
        values = [((r * 31 + 127) // 255 << 11) | ((g * 31 + 127) // 255 << 6) |
                  ((b * 31 + 127) // 255 << 1) | int(a >= 128) for r, g, b, a in pixels]
        px.data = struct.pack(f.e + f"{len(values)}H", *values) + px.data[2 * iw * ih:]
    else:
        raise ModelError(tr('cannot write this image format'))


def same_texture(a: str, b: str) -> bool:
    """Two names of one texture («Tiny_GAME.png» in the .igb, «Tiny_GAME» in the glTF)."""
    def stem(n: str) -> str:
        n = n.replace("\\", "/").rsplit("/", 1)[-1]
        return n.rsplit(".", 1)[0].lower() if "." in n else n.lower()
    return stem(a) == stem(b)


def match_meshes(model: Model, other: Model) -> list[tuple[Mesh, Mesh]]:
    """The other model's meshes on this one's, by name (Blender's «.001» endings left out)."""
    def base(name: str) -> str:
        name = name.split("|")[-1]
        while len(name) > 4 and name[-4] == "." and name[-3:].isdigit():
            name = name[:-4]
        return name
    by = {}
    for m in other.meshes:
        by.setdefault(base(m.name), []).append(m)
    out = []
    for m in model.meshes:
        cands = by.get(base(m.name)) or []
        if cands:
            out.append((m, cands.pop(0)))
    return out


def merge(meshes: list[Mesh]) -> Mesh:
    """Several pieces (a mesh Blender split by material) as one."""
    if len(meshes) == 1:
        return meshes[0]
    out = Mesh(meshes[0].name)
    out.nrm = [] if all(m.nrm for m in meshes) else None
    out.col = [] if any(m.col for m in meshes) else None
    out.uv = [] if all(m.uv for m in meshes) else None
    out.weights = [] if all(m.weights is not None for m in meshes) else None
    for m in meshes:
        base = len(out.pos)
        out.pos += m.pos
        if out.nrm is not None:
            out.nrm += m.nrm
        if out.col is not None:
            out.col += m.col or [NEUTRAL] * len(m.pos)
        if out.uv is not None:
            out.uv += m.uv
        if out.weights is not None:
            out.weights += m.weights
        out.tris += [(a + base, b + base, c + base) for a, b, c in m.tris]
    out.texture = meshes[0].texture
    return out


def unique_names(names: list[str]) -> list[str]:
    """Names made different as the glTF export makes them (a second «Pant_BackR01» is «Pant_BackR01.1»)."""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}.{seen[n] - 1}")
    return out


def bones_onto(model: Model, other: Model) -> dict[int, int]:
    """The other model's bones on this one's, by name (case aside, a «Armature|» prefix aside); a bone this one has
    not: its nearest parent that it has."""
    def key(n: str) -> str:
        return n.split("|")[-1].split(":")[-1].lower()
    by = {key(n): i for i, n in enumerate(unique_names([b.name for b in model.bones]))}
    out = {}
    for i, b in enumerate(other.bones):
        k, j = by.get(key(b.name)), i
        while k is None and j >= 0:
            j = other.bones[j].parent
            k = by.get(key(other.bones[j].name)) if j >= 0 else None
        out[i] = k if k is not None else 0
    return out


def apply(model: Model, other: Model) -> tuple[bytes, list[str]]:
    """The .igb of model (read from it) with the meshes and textures of other in place of its own (by name): the
    file's bytes and what was done. other's bones are matched to the file's by name: the skeleton stays the file's."""
    if model.igb is None:
        raise ModelError(tr('the model did not come from an IGB'))
    f = igb_file.IGB(model.igb.write())  # a copy to change
    notes: list[str] = []
    onto = bones_onto(model, other)
    pairs = match_meshes(model, other)
    done = set()
    for old, new in pairs:
        nm = new
        if nm.weights is not None:
            nm = _rebound(nm, onto)
        write_mesh(f, model, old, nm, notes)
        if nm.shapes:
            notes.append(tr('{0}: {1} blendshapes not written (unsupported by the game IGB)', old.name, len(nm.shapes)))
        done.add(old.name)

    for m in model.meshes:
        if m.name not in done:
            notes.append(tr('{0}: not found in the imported file; left unchanged', m.name))
    for t in other.textures:
        if not t.changed:
            continue
        mine = next((k for k in model.textures if same_texture(k.name, t.name)), None)
        if mine is None or mine.image < 0:
            notes.append(tr('texture {0}: no matching name in the file', t.name))
            continue
        encode_picture(f, f.objects[mine.image], t.w, t.h, t.rgba)
        notes.append(tr('texture {0}: replaced', t.name))
    return f.write(), notes


def _rebound(m: Mesh, onto: dict[int, int]) -> Mesh:
    out = Mesh(m.name)
    out.__dict__.update(m.__dict__)
    out.weights = [[(onto.get(b, 0), w) for b, w in vw] for vw in m.weights]
    return out


def with_textures(model: Model, changes: dict[int, tuple[int, int, bytes]]) -> tuple[bytes, list[str]]:
    """The .igb with some of its textures' pictures replaced (index into model.textures -> w, h, rgba)."""
    f = igb_file.IGB(model.igb.write())
    notes = []
    for k, (w, h, rgba) in changes.items():
        t = model.textures[k]
        encode_picture(f, f.objects[t.image], w, h, rgba)
        notes.append(tr('texture {0}: replaced', t.name))
    return f.write(), notes


# Looking.

def write_obj(model: Model, path: Path) -> None:
    lines, base = [f"# {model.source}"], 1
    for m in model.meshes:
        lines.append(f"o {m.name}")
        place = model.bones[m.bone].bind if 0 <= m.bone < len(model.bones) else (IDENTITY if m.skinned else m.matrix)
        lines += [f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in (point(q, place) for q in m.pos)]
        uv = m.uv or [(0.0, 0.0)] * len(m.pos)
        lines += [f"vt {t[0]:.6f} {1 - t[1]:.6f}" for t in uv]
        lines += [f"f {a + base}/{a + base} {b + base}/{b + base} {c + base}/{c + base}" for a, b, c in m.tris]
        base += len(m.pos)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe(model: Model) -> list[str]:
    out = [tr('{0} bones, {1} meshes, {2} textures, {3} animations', len(model.bones), len(model.meshes), len(model.textures), len(model.clips))]
    for m in model.meshes:
        kind = "skin" if m.skinned else ('rigid on bone' if m.bone >= 0 else 'rigid')
        out.append(tr('  {0}: {1} vertices, {2} triangles, {3}, LOD {4}, texture {5}', m.name, len(m.pos), len(m.tris), kind, m.lod, model.textures[m.texture].name if m.texture >= 0 else '-'))
    for c in model.clips:
        out.append(tr('  animation {0}: {1:.2f} s', c.name, c.length))
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    model = from_igb(Path(sys.argv[1]).read_bytes(), Path(sys.argv[1]).name)
    print("\n".join(describe(model)))
    if len(sys.argv) > 3 and sys.argv[2] == "--obj":
        write_obj(model, Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
