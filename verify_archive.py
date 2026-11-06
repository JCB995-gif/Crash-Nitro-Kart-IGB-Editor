"""Validate every IGB in a PS2 or Xbox archive without modifying game data.

python tools/cnk_igb_editor/verify_archive.py ASSETS.GFC [--characters] [--report report.json]
"""
import argparse
import collections
import json
import math
from pathlib import Path

import cnk_gob
import cnk_assets
import igb_model


def verify(path, characters=False, contains="", contents=False):
    archive = cnk_gob.Gob(path)
    totals = collections.Counter()
    failures, notes = [], []
    for entry in archive.igbs():
        if contains.casefold() not in entry.path.casefold():
            continue
        if characters and not any(p in entry.path for p in ("/chars/", "/charmodels/", "/karts/")):
            continue
        totals["files"] += 1
        try:
            data = archive.read(entry)
            model = igb_model.from_igb(data, entry.path, archive.platform)
            if model.igb.write() != data:
                raise ValueError("IGB round trip changed original bytes")
            f = model.igb
            skeleton = igb_model.main_skeleton(f)
            if skeleton is not None:
                expected = len(f.items(f.gref(skeleton, igb_model.SK_BONES)))
                if expected != len(model.bones):
                    raise ValueError(f"Skeleton lost bones: {len(model.bones)} of {expected}")
            expected = len(f.of("igAnimation"))
            actual = sum(isinstance(c, igb_model.IgbClip) for c in model.clips)
            if expected != actual:
                raise ValueError(f"Lost skeletal animations: {actual} of {expected}")
            for clip in (c for c in model.clips if isinstance(c, igb_model.IgbClip)):
                anim = f.objects[clip.object_index]
                if len(clip.tracks) != len(f.items(f.gref(anim, igb_model.AN_TRACKS))):
                    raise ValueError(f"{clip.name}: lost animation tracks")
            for mesh in model.meshes:
                if any(not math.isfinite(v) for p in mesh.pos for v in p):
                    raise ValueError(f"{mesh.name}: non-finite vertex")
                if any(not 0 <= i < len(mesh.pos) for t in mesh.tris for i in t):
                    raise ValueError(f"{mesh.name}: invalid triangle")
                if mesh.weights and any(not 0 <= b < len(model.bones) or not math.isfinite(w)
                                        for row in mesh.weights for b, w in row):
                    raise ValueError(f"{mesh.name}: invalid bone weights")
            for texture in model.textures:
                if len(texture.rgba) != texture.w * texture.h * 4:
                    raise ValueError(f"{texture.name}: invalid RGBA length")
            for clip in model.clips:
                for t in (0, clip.length * .5, clip.length):
                    pose = clip.pose(t)
                    if len(pose) != len(model.bones) or any(not math.isfinite(v) for m in pose for v in m):
                        raise ValueError(f"{clip.name}: invalid animation pose")
                if hasattr(clip, "_cache"):
                    clip._cache.clear()
            if model.scene_motion:
                motion = model.scene_motion
                for index, phase in motion.channels:
                    for t in (0, motion.lengths[phase] * .5, motion.lengths[phase]):
                        if not all(math.isfinite(v) for v in motion.local(index, phase, t)):
                            raise ValueError(f"transform #{index}, phase {phase}: non-finite matrix")
            totals.update(meshes=len(model.meshes), bones=len(model.bones), textures=len(model.textures),
                          animations=len(model.clips), triangles=sum(len(m.tris) for m in model.meshes))
            if model.notes:
                notes.append({"file": entry.path, "notes": model.notes})
        except Exception as ex:
            failures.append({"file": entry.path, "error": f"{type(ex).__name__}: {ex}"})
            print(f"FAIL {entry.path}: {ex}", flush=True)
        if totals["files"] % 25 == 0:
            print(f"Checked {totals['files']} files; {len(failures)} failures", flush=True)
    result = {"archive": str(path), "totals": dict(totals), "failures": failures, "notes": notes}
    if contents:
        counts, types = collections.Counter(), collections.Counter()
        for block in archive.check():
            failures.append({"file": "GOB", "error": f"Checksum mismatch: {block}"})
        for entry in archive.files:
            if contains.casefold() not in entry.path.casefold():
                continue
            try:
                data = archive.read(entry)
                counts.update(files=1, bytes=len(data))
                suffix = Path(entry.path).suffix.lower()
                types[suffix] += 1
                if suffix in (".png", ".tga"):
                    w, h, rgba = cnk_assets.image(data, entry.path)
                    if len(rgba) != w * h * 4:
                        raise ValueError("Invalid image size")
                    counts["images"] += 1
            except Exception as ex:
                failures.append({"file": entry.path, "error": f"{type(ex).__name__}: {ex}"})
        result["contents"] = {**dict(counts), "types": dict(types), "blocks_checked": len(archive.blocks)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--characters", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--filter", default="", help="Only paths containing this text")
    parser.add_argument("--contents", action="store_true", help="Also read every archive entry, decode PNG/TGA and check block checksums")
    args = parser.parse_args()
    result = verify(args.archive, args.characters, args.filter, args.contents)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
