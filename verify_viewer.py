"""Windows OpenGL smoke test; isolated from the user's editor settings.

python tools/cnk_igb_editor/verify_viewer.py ASSETS.GFC [ASSETS.GFC ...] --report report.json
Use --all for scenery/effects too; --previews DIR saves a few framebuffer images.
"""
import argparse
import collections
import ctypes
import json
from pathlib import Path
import sys
import traceback

import cnk_gob
import cnk_igb_gui as gui
import igb_file
import igb_model
import png_util


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--previews", type=Path)
    parser.add_argument("--filter", default="", help="Only paths containing this text")
    parser.add_argument("--samples", type=int, default=3, help="Frames per animation, including its endpoints")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("The viewer uses Windows OpenGL")
    gui.load_settings = lambda: {}
    gui.save_settings = lambda settings: None
    app = gui.Viewer(None)
    app.attributes("-alpha", 0.0)
    errors, totals = [], collections.Counter()
    active_file = "initialization"
    app.report_callback_exception = lambda *ex: errors.append(
        {"file": active_file, "error": "".join(traceback.format_exception(*ex))})
    try:
        app.update()
        if app.view.sh is None:
            raise RuntimeError(app.status_label.cget("text"))
        get_error = app.view.gl.lib.glGetError
        get_error.restype = ctypes.c_uint

        def draw():
            app.view.draw()
            code = get_error()
            if code:
                raise RuntimeError(f"OpenGL error {code:#x}")
            totals["frames"] += 1

        for path in args.archives:
            archive = cnk_gob.Gob(path)
            for entry in archive.igbs():
                if args.filter.casefold() not in entry.path.casefold():
                    continue
                if not args.all and not args.filter and not any(p in entry.path for p in ("/chars/", "/charmodels/", "/karts/")):
                    continue
                active_file = entry.path
                try:
                    model = igb_model.from_igb(archive.read(entry), entry.path, archive.platform)
                    app.show(model)
                    app.update()
                    if not model.meshes and model.textures:
                        if not app.view.image_panel.winfo_ismapped() or app.view.preview_photo is None:
                            raise RuntimeError("Texture-only file did not open in the image viewer")
                        if app.tabs.select() != str(app.texture_tab):
                            raise RuntimeError("Texture-only file did not select the texture tab")
                    draw()
                    best = set(app.hidden)
                    app.hidden.clear()  # include every LOD, rigid attachment and shadow mesh
                    draw()
                    app.hidden = best
                    if args.previews and entry.path.endswith(("/chars/crash.igb", "/charmodels/tiny.igb")):
                        args.previews.mkdir(parents=True, exist_ok=True)
                        shot = app.view.snapshot(800, 600)
                        if shot is None:
                            raise RuntimeError("Framebuffer snapshot unavailable")
                        filename = entry.path.replace("/", "_").replace(".igb", ".png")
                        (args.previews / filename).write_bytes(png_util.png_bytes(*shot))
                    for i, clip in enumerate(model.clips):
                        app.clip = i
                        samples = max(2, args.samples)
                        for fraction in (k / (samples - 1) for k in range(samples)):
                            app.time = clip.length * fraction
                            draw()
                        totals["animations"] += 1
                    if app.face_indices:
                        app.face_picked()
                        app.face_vars[0].set(15)
                        app.face_changed()
                        draw()
                        app.zero_face()
                    if model.igb.objects:
                        obj = next((o for o in model.igb.objects if isinstance(o, igb_file.Obj)), None)
                        mem = next((o for o in model.igb.objects if isinstance(o, igb_file.Mem)), None)
                        if obj:
                            app.inspector.show_object(obj.index)
                        if mem:
                            app.inspector.show_object(mem.index)
                            app.inspector.memory_page(1)
                    totals["files"] += 1
                    if totals["files"] % 25 == 0:
                        print(f"Rendered {totals['files']} files / {totals['frames']} frames; {len(errors)} errors",
                              flush=True)
                except Exception:
                    errors.append({"file": active_file, "error": traceback.format_exc()})
        result = {"totals": dict(totals), "errors": errors}
        if args.report:
            args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return int(bool(errors))
    finally:
        app.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
