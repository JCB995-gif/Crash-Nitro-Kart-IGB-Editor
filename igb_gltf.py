#!/usr/bin/env python3
"""An igb_model.Model to glTF 2.0 for Blender and back: the skeleton as joints (its bind pose as rest), the meshes
with their textures (skins weighted to the joints, rigid parts placed where they are, a part that moves with a bone
skinned wholly to it), blend shapes as morph targets, the animations sampled 30 times a second. A .glb or a .gltf
(its data inside) out; a .glb or .gltf (data inside or in files beside it) in, from this tool or from Blender.

The .igb is Z up, glTF Y up: a point (x, y, z) goes out as (x, z, -y) and comes back the same way, a turn and not a
mirror, so the triangles keep their winding. The meshes go out in the model's space (a rigid part's matrix put into
its vertices) and come back to it; each keeps the name of its igGeometry, which is how the import finds it again.

    igb_gltf.py export <file.igb> <out.glb|out.gltf>
    igb_gltf.py import <file.igb> <in.glb|in.gltf> <out.igb>     the meshes of the glTF in place of the file's
"""
from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gltf_util  # noqa: E402
import igb_model  # noqa: E402
import png_util  # noqa: E402
from igb_model import IDENTITY, inverse, mul  # noqa: E402

FPS = 30.0
C = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]   # .igb -> glTF
CI = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # glTF -> .igb


class GltfError(ValueError):
    pass


def out_p(p) -> tuple:
    return (p[0], p[2], -p[1])


def in_p(p) -> tuple:
    return (p[0], -p[2], p[1])


def out_m(m: list) -> list:
    return mul(mul(CI, m), C)


def in_m(m: list) -> list:
    return mul(mul(C, m), CI)


# Out.

def export(model: igb_model.Model, meshes: list[igb_model.Mesh] | None = None, clips: bool = True,
           name: str = "") -> tuple[dict, bytes]:
    """The glTF's JSON and its binary buffer."""
    meshes = [m for m in (model.meshes if meshes is None else meshes) if m.pos and m.tris]
    buf = gltf_util.Buffer()
    bones = model.bones
    names = igb_model.unique_names([b.name for b in bones])
    nodes: list[dict] = []
    for i, b in enumerate(bones):
        local = b.bind if b.parent < 0 else mul(b.bind, inverse(bones[b.parent].bind))
        t, q, s = gltf_util.trs(out_m(local))
        node = {"name": names[i], "translation": t, "rotation": q}
        if any(abs(v - 1.0) > 1e-5 for v in s):
            node["scale"] = s
        kids = [k for k, c in enumerate(bones) if c.parent == i]
        if kids:
            node["children"] = kids
        nodes.append(node)
    roots = [i for i, b in enumerate(bones) if b.parent < 0]
    gltf: dict = {"asset": {"version": "2.0", "generator": "CrashWOC NuWorld igb_gltf"}, "scene": 0,
                  "extras": {"igb": {"source": model.source, "fps": FPS}}}
    if bones:
        ibm = [v for b in bones for v in inverse(out_m(b.bind))]
        gltf["skins"] = [{"name": name or "skeleton", "joints": list(range(len(bones))),
                          "skeleton": roots[0] if roots else 0,
                          "inverseBindMatrices": buf.add("f", ibm, "MAT4", len(bones), 5126)}]
    # Materials: one a texture (and one for none).
    images, textures, materials = [], [], []
    mat_of: dict[int, int] = {}

    def material(ti: int) -> int:
        if ti in mat_of:
            return mat_of[ti]
        mat = {"name": "untextured" if ti < 0 else Path(model.textures[ti].name).stem, "doubleSided": True,
               "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}}
        if 0 <= ti < len(model.textures):
            t = model.textures[ti]
            images.append({"name": Path(t.name).stem, "mimeType": "image/png",
                           "bufferView": buf.raw(png_util.png_bytes(t.w, t.h, t.rgba))})
            textures.append({"source": len(images) - 1, "sampler": 0})
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(textures) - 1}
            if any(a < 250 for a in t.rgba[3::4]):
                mat["alphaMode"], mat["alphaCutoff"] = "MASK", 0.5
        materials.append(mat)
        mat_of[ti] = len(materials) - 1
        return mat_of[ti]

    gl_meshes = []
    mesh_nodes = []
    for m in meshes:
        if m.skinned:
            place = IDENTITY
        elif 0 <= m.bone < len(bones):
            place = bones[m.bone].bind
        else:
            place = m.matrix
        place = igb_model.plain(place)
        pos = [out_p(p if place is IDENTITY else igb_model.point(p, place)) for p in m.pos]
        attrs = {"POSITION": buf.add("f", [v for p in pos for v in p], "VEC3", len(pos), 5126, 34962, minmax=True)}
        if m.nrm:
            nrm = [out_p(n if place is IDENTITY else igb_model.normalised(igb_model.direction(n, place)))
                   for n in m.nrm]
            attrs["NORMAL"] = buf.add("f", [v for n in nrm for v in n], "VEC3", len(nrm), 5126, 34962)
        if m.uv:
            attrs["TEXCOORD_0"] = buf.add("f", [v for t in m.uv for v in t], "VEC2", len(m.uv), 5126, 34962)
        if m.col:
            attrs["COLOR_0"] = buf.add("f", [c / 255.0 for col in m.col for c in col], "VEC4", len(m.col), 5126,
                                       34962)
        skinned = bool(bones) and (m.skinned or 0 <= m.bone < len(bones))
        if skinned:
            joints, weights = [], []
            for i in range(len(m.pos)):
                vw = m.weights[i] if m.skinned else [(m.bone, 1.0)]
                top = sorted(vw, key=lambda x: -x[1])[:4]
                total = sum(w for _, w in top) or 1.0
                if abs(total - 1.0) < 1e-6:  # as they are (back unchanged)
                    total = 1.0
                top += [(0, 0.0)] * (4 - len(top))
                joints += [j for j, _ in top]
                weights += [w / total for _, w in top]
            attrs["JOINTS_0"] = buf.add("H", joints, "VEC4", len(m.pos), 5123, 34962)
            attrs["WEIGHTS_0"] = buf.add("f", weights, "VEC4", len(m.pos), 5126, 34962)
        prim = {"attributes": attrs, "material": material(m.texture),
                "indices": buf.add("I", [i for t in m.tris for i in t], "SCALAR", 3 * len(m.tris), 5125, 34963)}
        extras = {"igb": {"lod": m.lod, "skin": m.skin_name, "colorScale": m.color_scale}}
        if m.strips is not None and sum(m.strips) == len(m.pos):
            extras["igb"]["strips"] = m.strips  # (back unchanged, the vertices keep their strips)
        if m.shapes:
            prim["targets"] = [{"POSITION": buf.add("f", [v for d in deltas for v in out_p(d)], "VEC3",
                                                    len(deltas), 5126, 34962, minmax=True)}
                               for _, deltas in m.shapes]
            extras["targetNames"] = [n for n, _ in m.shapes]
        mesh = {"name": m.name, "primitives": [prim], "extras": extras}
        if m.shapes:
            mesh["weights"] = [0.0] * len(m.shapes)
        gl_meshes.append(mesh)
        node = {"name": m.name, "mesh": len(gl_meshes) - 1}
        if skinned:
            node["skin"] = 0
        nodes.append(node)
        mesh_nodes.append(len(nodes) - 1)
    gltf["nodes"] = nodes
    gltf["scenes"] = [{"name": name or model.source or "igb", "nodes": roots + mesh_nodes}]
    if gl_meshes:
        gltf["meshes"] = gl_meshes
        gltf["materials"] = materials
    if images:
        gltf["images"], gltf["textures"] = images, textures
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    if clips and bones and model.clips:
        gltf["animations"] = [anim for c in model.clips if (anim := clip_out(c, bones, buf))]
    gltf["buffers"] = [{"byteLength": len(buf.data)}]
    gltf["bufferViews"] = buf.views
    gltf["accessors"] = buf.accessors
    return gltf, bytes(buf.data)


def clip_out(clip: igb_model.Clip, bones: list[igb_model.Bone], buf: gltf_util.Buffer) -> dict | None:
    n = max(2, int(round(clip.length * FPS)) + 1)
    times = [min(clip.length, k / FPS) for k in range(n)]
    if times[-1] < clip.length - 1e-6:
        times.append(clip.length)
    samples = [[] for _ in bones]
    for t in times:
        world = clip.pose(t)
        for i, b in enumerate(bones):
            local = world[i] if b.parent < 0 else mul(world[i], inverse(world[b.parent]))
            samples[i].append(gltf_util.trs(out_m(local)))
    tacc = buf.add("f", times, "SCALAR", len(times), 5126, minmax=True)
    channels, samplers = [], []
    for i, smp in enumerate(samples):
        prev, quats = None, []
        for _, q, _ in smp:
            if prev and sum(a * b for a, b in zip(prev, q)) < 0:
                q = [-v for v in q]
            quats.append(q)
            prev = q
        series = [("translation", [v for t, _, _ in smp for v in t], "VEC3"),
                  ("rotation", [v for q in quats for v in q], "VEC4")]
        if any(abs(v - 1.0) > 1e-4 for _, _, s in smp for v in s):
            series.append(("scale", [v for _, _, s in smp for v in s], "VEC3"))
        for path, values, kind in series:
            samplers.append({"input": tacc, "output": buf.add("f", values, kind, len(times), 5126),
                             "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": i, "path": path}})
    return {"name": clip.name, "channels": channels, "samplers": samplers} if channels else None


def glb_bytes(gltf: dict, data: bytes) -> bytes:
    gltf = dict(gltf)
    gltf["buffers"] = [{"byteLength": len(data)}]
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    data = data + b"\0" * (-len(data) % 4)
    total = 12 + 8 + len(js) + 8 + len(data)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js +
            struct.pack("<II", len(data), 0x004E4942) + data)


def save(model: igb_model.Model, path: Path, meshes=None, clips: bool = True) -> None:
    gltf, data = export(model, meshes, clips, Path(model.source or path.stem).stem)
    if path.suffix.lower() == ".glb":
        path.write_bytes(glb_bytes(gltf, data))
    else:
        gltf["buffers"] = [{"byteLength": len(data),
                            "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}]
        path.write_text(json.dumps(gltf, indent=1), encoding="utf-8")


# In.

def accessor(gltf: dict, buffers: list[bytes], index: int) -> list[list[float]]:
    """An accessor's values (gltf_util's reader, and a sparse accessor's replaced values on top)."""
    acc = gltf["accessors"][index]
    out = gltf_util.accessor(gltf, buffers, index)
    sp = acc.get("sparse")
    if sp:
        n = sp["count"]
        width = gltf_util.WIDTH[acc["type"]]
        fmt, norm = gltf_util.COMPONENTS[acc["componentType"]]
        iv = gltf["bufferViews"][sp["indices"]["bufferView"]]
        ifmt = gltf_util.COMPONENTS[sp["indices"]["componentType"]][0]
        ibase = iv.get("byteOffset", 0) + sp["indices"].get("byteOffset", 0)
        idx = struct.unpack_from(f"<{n}{ifmt}", buffers[iv["buffer"]], ibase)
        vv = gltf["bufferViews"][sp["values"]["bufferView"]]
        vbase = vv.get("byteOffset", 0) + sp["values"].get("byteOffset", 0)
        size = struct.calcsize(fmt) * width
        for k, i in enumerate(idx):
            v = list(struct.unpack_from(f"<{width}{fmt}", buffers[vv["buffer"]], vbase + k * size))
            if norm and acc.get("normalized", fmt != "I"):
                v = [max(x / norm, -1.0) for x in v]
            out[i] = v
    return out


def raw(gltf: dict, buffers: list[bytes], index: int) -> list[list[int]]:
    """An accessor of whole numbers as they are (joints, indices: never scaled to 0-1)."""
    acc = dict(gltf["accessors"][index])
    acc["normalized"] = False
    copy = dict(gltf)
    copy["accessors"] = list(gltf["accessors"])
    copy["accessors"][index] = acc
    return [[int(round(x)) for x in v] for v in accessor(copy, buffers, index)]


def node_local(node: dict) -> list:
    if "matrix" in node:
        return [float(v) for v in node["matrix"]]
    return gltf_util.from_trs(node.get("translation", [0.0, 0.0, 0.0]), node.get("rotation", [0.0, 0.0, 0.0, 1.0]),
                              node.get("scale", [1.0, 1.0, 1.0]))


def parents_of(gltf: dict) -> list[int]:
    par = [-1] * len(gltf.get("nodes", []))
    for i, n in enumerate(gltf.get("nodes", [])):
        for k in n.get("children", []):
            if 0 <= k < len(par):
                par[k] = i
    return par


def worlds(gltf: dict, locals_: list[list], par: list[int]) -> list[list]:
    out: list = [None] * len(locals_)

    def world(i: int, depth: int = 0) -> list:
        if out[i] is None:
            p = par[i]
            out[i] = locals_[i] if p < 0 or depth > 256 else mul(locals_[i], world(p, depth + 1))
        return out[i]
    for i in range(len(locals_)):
        world(i)
    return out


def image_bytes(gltf: dict, buffers: list[bytes], path: Path, img: dict) -> bytes | None:
    if "bufferView" in img:
        v = gltf["bufferViews"][img["bufferView"]]
        off = v.get("byteOffset", 0)
        return buffers[v["buffer"]][off:off + v["byteLength"]]
    uri = img.get("uri", "")
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    if uri:
        from urllib.parse import unquote
        p = path.parent / unquote(uri)
        return p.read_bytes() if p.is_file() else None
    return None


def rgba_of(data: bytes, name: str) -> tuple[int, int, bytes] | None:
    if not data or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None  # (a JPEG: no decoder in the standard library)
    w, h, pixels, palette = png_util.png_decode(data, name)
    if palette is not None:
        pixels = [palette[i] if i < len(palette) else (0, 0, 0, 255) for i in pixels]
    return w, h, b"".join(bytes(p if len(p) == 4 else (*p[:3], 255)) for p in pixels)


class GltfClip(igb_model.Clip):
    """A glTF animation over the model's bones: every node's rest or animated local, through the hierarchy."""

    def __init__(self, gltf: dict, buffers: list[bytes], anim: dict, rest: list[list], par: list[int],
                 joint_nodes: list[int], name: str):
        self.tracks: dict[int, dict] = {}
        end = 0.0
        for ch in anim.get("channels", []):
            tgt = ch.get("target", {})
            node, path = tgt.get("node"), tgt.get("path")
            if node is None or path not in ("translation", "rotation", "scale"):
                continue
            tr = gltf_util.Track(gltf, buffers, anim["samplers"][ch["sampler"]], path == "rotation")
            self.tracks.setdefault(node, {})[path] = tr
            end = max(end, tr.times[-1] if tr.times else 0.0)
        super().__init__(name, end or 1.0)
        self.nodes = gltf.get("nodes", [])
        self.rest, self.par, self.joint_nodes = rest, par, joint_nodes
        self._cache: dict[int, list] = {}

    def pose(self, t: float) -> list[list]:
        key = round(t * 1000)
        if key in self._cache:
            return self._cache[key]
        locs = list(self.rest)
        for node, chans in self.tracks.items():
            n = self.nodes[node]
            tt = chans["translation"](t) if "translation" in chans else n.get("translation", [0.0, 0.0, 0.0])
            q = chans["rotation"](t) if "rotation" in chans else n.get("rotation", [0.0, 0.0, 0.0, 1.0])
            s = chans["scale"](t) if "scale" in chans else n.get("scale", [1.0, 1.0, 1.0])
            if "matrix" in n and not chans:
                continue
            locs[node] = gltf_util.from_trs(list(tt), list(q), list(s))
        w = worlds({}, locs, self.par)
        out = [in_m(w[j]) for j in self.joint_nodes]
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[key] = out
        return out


def load(path: Path) -> igb_model.Model:
    """A .gltf or .glb as a model in the .igb's space."""
    try:
        gltf, buffers = gltf_util.read_gltf(path)
    except (OSError, ValueError, KeyError) as e:
        raise GltfError(f"glTF illeggibile: {e}") from None
    model = igb_model.Model()
    model.source = path.name
    nodes = gltf.get("nodes", [])
    par = parents_of(gltf)
    rest = [node_local(n) for n in nodes]
    world = worlds(gltf, rest, par)
    # The skeleton: the skin most meshes use, its joints with parents first.
    used = [n["skin"] for n in nodes if "mesh" in n and "skin" in n]
    skins = gltf.get("skins", [])
    skin_i = max(set(used), key=used.count) if used else (0 if skins else None)
    joint_nodes: list[int] = []
    bone_of_joint: dict[int, int] = {}
    if skin_i is not None and skin_i < len(skins):
        skin = skins[skin_i]
        joints = list(skin.get("joints", []))
        ibms = accessor(gltf, buffers, skin["inverseBindMatrices"]) if "inverseBindMatrices" in skin else None
        jset = set(joints)

        def joint_parent(nd: int) -> int:
            p = par[nd]
            while p >= 0 and p not in jset:
                p = par[p]
            return p
        order, placed = [], set()

        def place(nd: int, depth: int = 0) -> None:
            if nd in placed or depth > 256:
                return
            p = joint_parent(nd)
            if p >= 0:
                place(p, depth + 1)
            placed.add(nd)
            order.append(nd)
        for nd in joints:
            place(nd)
        index_of = {nd: k for k, nd in enumerate(order)}
        for nd in order:
            j = joints.index(nd)
            bind_g = inverse(ibms[j]) if ibms else world[nd]
            p = joint_parent(nd)
            model.bones.append(igb_model.Bone(nodes[nd].get("name") or f"joint{nd}", index_of.get(p, -1),
                                              in_m(bind_g)))
            bone_of_joint[j] = index_of[nd]
        joint_nodes = order
    # Textures.
    tex_of: dict[int, int] = {}

    def texture(mi) -> int:
        if mi is None or mi >= len(gltf.get("materials", [])):
            return -1
        info = gltf["materials"][mi].get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if not info:
            return -1
        ti = info.get("index", -1)
        if ti in tex_of:
            return tex_of[ti]
        tex_of[ti] = -1
        try:
            src = gltf["textures"][ti].get("source")
            img = gltf["images"][src]
            nm = img.get("name") or Path(img.get("uri", "") or f"texture{src}").stem
            got = rgba_of(image_bytes(gltf, buffers, path, img) or b"", nm)
        except (IndexError, KeyError, ValueError, OSError):
            got = None
        if got:
            model.textures.append(igb_model.Texture(nm, *got))
            tex_of[ti] = len(model.textures) - 1
        return tex_of[ti]
    # The meshes: a node's primitives as one mesh.
    scene_nodes = set()
    scenes = gltf.get("scenes", [])
    roots = scenes[gltf.get("scene", 0)]["nodes"] if scenes else [i for i, p in enumerate(par) if p < 0]
    stack = list(roots)
    while stack:
        i = stack.pop()
        if i in scene_nodes or i >= len(nodes):
            continue
        scene_nodes.add(i)
        stack += nodes[i].get("children", [])
    for ni in sorted(scene_nodes):
        n = nodes[ni]
        if "mesh" not in n:
            continue
        gm = gltf["meshes"][n["mesh"]]
        skinned = "skin" in n and n["skin"] == skin_i and model.bones
        pieces = []
        names = (gm.get("extras") or {}).get("targetNames") or []
        for prim in gm.get("primitives", []):
            mode = prim.get("mode", 4)
            if mode not in (4, 5, 6) or "POSITION" not in prim.get("attributes", {}):
                continue
            a = prim["attributes"]
            m = igb_model.Mesh(n.get("name") or gm.get("name") or f"mesh{ni}")
            pos = accessor(gltf, buffers, a["POSITION"])
            place = IDENTITY if skinned else igb_model.plain(world[ni])
            m.pos = [in_p(p if place is IDENTITY else igb_model.point(p, place)) for p in pos]
            if "NORMAL" in a:
                m.nrm = [in_p(v if place is IDENTITY else igb_model.normalised(igb_model.direction(v, place)))
                         for v in accessor(gltf, buffers, a["NORMAL"])]
            if "TEXCOORD_0" in a:
                m.uv = [tuple(v[:2]) for v in accessor(gltf, buffers, a["TEXCOORD_0"])]
            if "COLOR_0" in a:
                m.col = [tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in (v + [1.0])[:4])
                         for v in accessor(gltf, buffers, a["COLOR_0"])]
            if skinned and "JOINTS_0" in a and "WEIGHTS_0" in a:
                js = raw(gltf, buffers, a["JOINTS_0"])
                ws = accessor(gltf, buffers, a["WEIGHTS_0"])
                if "JOINTS_1" in a and "WEIGHTS_1" in a:
                    js = [x + y for x, y in zip(js, raw(gltf, buffers, a["JOINTS_1"]))]
                    ws = [x + y for x, y in zip(ws, accessor(gltf, buffers, a["WEIGHTS_1"]))]
                m.weights = [[(bone_of_joint.get(int(j), 0), w) for j, w in zip(jv, wv) if w >= 0] or [(0, 1.0)]
                             for jv, wv in zip(js, ws)]
            if "indices" in prim:
                idx = [v[0] for v in raw(gltf, buffers, prim["indices"])]
            else:
                idx = list(range(len(pos)))
            if mode == 4:
                m.tris = [tuple(idx[k:k + 3]) for k in range(0, len(idx) - 2, 3)]
            elif mode == 5:
                m.tris = [(idx[k], idx[k + 1], idx[k + 2]) if k % 2 == 0 else (idx[k], idx[k + 2], idx[k + 1])
                          for k in range(len(idx) - 2)]
            else:
                m.tris = [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
            m.tris = [t for t in m.tris if len(set(t)) == 3]
            for k, tgt in enumerate(prim.get("targets", [])):
                if "POSITION" in tgt:
                    d = [in_p(igb_model.direction(v, place)) for v in accessor(gltf, buffers, tgt["POSITION"])]
                    m.shapes.append((names[k] if k < len(names) else f"shape_{k:02d}", d))
            m.texture = texture(prim.get("material"))
            pieces.append(m)
        if not pieces:
            continue
        mesh = igb_model.merge(pieces) if len(pieces) > 1 else pieces[0]
        if len(pieces) > 1 and all(len(p.shapes) == len(pieces[0].shapes) for p in pieces):
            mesh.shapes = [(nm, [d for p in pieces for d in p.shapes[k][1]]) for k, (nm, _) in
                           enumerate(pieces[0].shapes)]
        if not skinned:
            mesh.weights = None
        info = (gm.get("extras") or {}).get("igb") or {}
        mesh.lod = int(info.get("lod", 0) or 0)
        mesh.color_scale = float(info.get("colorScale", 2.0))
        strips = info.get("strips")
        if len(pieces) == 1 and isinstance(strips, list) and sum(strips) == len(mesh.pos):
            mesh.strips = [int(v) for v in strips]
        mesh.default_weights = list(gm.get("weights") or [])
        model.meshes.append(mesh)
    if model.bones:
        rest_full = rest
        for i, a in enumerate(gltf.get("animations", [])):
            model.clips.append(GltfClip(gltf, buffers, a, rest_full, par, joint_nodes,
                                        a.get("name") or f"animazione {i + 1}"))
    return model


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    src = Path(sys.argv[2])
    model = igb_model.from_igb(src.read_bytes(), src.name)
    if cmd == "export":
        save(model, Path(sys.argv[3]), [m for m in model.meshes if m.lod == 0])
        print(f"{sys.argv[3]}: {len(model.bones)} joints, {len(model.meshes)} meshes, {len(model.clips)} animations")
    elif cmd == "import" and len(sys.argv) > 4:
        other = load(Path(sys.argv[3]))
        data, notes = igb_model.apply(model, other)
        Path(sys.argv[4]).write_bytes(data)
        print("\n".join(notes))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
