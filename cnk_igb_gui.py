#!/usr/bin/env python3
"""A small window over Crash Nitro Kart's .igb models: the model in 3D (textured, lit), its bones (picked in the
view or in their tree, the selected one's weights painted on the skin), its animations played, its blend shapes
moved with sliders; .igb files opened loose or straight from the game's ASSETS.GFC; the model out to glTF (Blender)
or OBJ, its textures to PNG; a glTF (edited in Blender) and PNG textures back into the .igb, saved as a new file.

Double-click CNK IGB Editor.pyw, or: cnk_igb_gui.py [file.igb | file.glb | file.gltf]
"""
from __future__ import annotations

from i18n import tr
import array
import base64
import ctypes
import json
import math
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnk_gob  # noqa: E402
import cnk_assets  # noqa: E402
import i18n  # noqa: E402
import igb_file  # noqa: E402
import igb_gltf  # noqa: E402
import igb_model  # noqa: E402
from igb_inspector import Inspector  # noqa: E402
import png_util  # noqa: E402
from gl_core import (GL, GL_BLEND, GL_COLOR_ARRAY, GL_COLOR_BUFFER_BIT, GL_CULL_FACE, GL_DEPTH_BUFFER_BIT,  # noqa
                     GL_DEPTH_TEST, GL_FILL, GL_FLOAT, GL_FRONT_AND_BACK, GL_LEQUAL, GL_LINE, GL_LINEAR, GL_LINES, GL_NEAREST,
                     GL_MODELVIEW, GL_PACK_ALIGNMENT, GL_PROJECTION, GL_REPEAT, GL_RGBA, GL_TEXTURE_2D,
                     GL_TEXTURE_COORD_ARRAY, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TRIANGLES, GL_UNPACK_ALIGNMENT,
                     GL_UNSIGNED_BYTE, GL_UNSIGNED_INT, GL_VERTEX_ARRAY, PIXELFORMATDESCRIPTOR)
from igb_model import IDENTITY  # noqa: E402

SETTINGS = Path.home() / ".cnk-igb-editor.json"  # (i18n reads its «language» from here too)
MAX_BONES = 128  # the shader's matrices
FOV = 0.8
GL_POINTS = 0
HEAT_HELP = 'Weights: blue 0, green 0.5, red 1'
NO_SHAPES = ('No blendshapes in this model.\n\nCrash Nitro Kart character expressions use facial bones (MouthJaw, Lower_Jaw, Eye_L/R, Eyebrow_L/R…), which can be inspected and posed in the Face tab and in animations.\n\nBlendshapes appear when imported from a glTF with Blender shape keys and are included in glTF exports; the game IGB format cannot store them.')
HELP = ("Open a Crash Nitro Kart IGB file, either loose or from the disc's ASSETS.GFC archive (Archive tab).\n\nBones: click a joint in the view or a name in the tree to display its skin weights (blue: zero, red: full). Animations: select a clip and press Play.\n\nExport glTF sends meshes, bones, weights, textures and animations to Blender (File > Import > glTF 2.0). After editing, export glTF with skinning and keep mesh and bone names. Open the original IGB here, choose “Import glTF…”, then “Save .igb…”. The original skeleton is preserved; vertices retain the strongest bone influences supported by the file.\n\nReplace textures with PNG files in the Texture tab. Images are converted to the original dimensions and colors.")


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> None:
    try:
        SETTINGS.write_text(json.dumps(settings, indent=1))
    except OSError:
        pass


class Shader:
    """The OpenGL 2.0 program: a skin through its bones' matrices (or a rigid mesh through one), lit from the
    camera, its texture, or the selected bone's weights as colours."""
    _sigs = {
        "glCreateShader": (ctypes.c_uint, [ctypes.c_uint]),
        "glShaderSource": (None, [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_void_p]),
        "glCompileShader": (None, [ctypes.c_uint]),
        "glGetShaderiv": (None, [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]),
        "glGetShaderInfoLog": (None, [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p]),
        "glCreateProgram": (ctypes.c_uint, []),
        "glAttachShader": (None, [ctypes.c_uint, ctypes.c_uint]),
        "glDeleteShader": (None, [ctypes.c_uint]),
        "glDeleteProgram": (None, [ctypes.c_uint]),
        "glBindAttribLocation": (None, [ctypes.c_uint, ctypes.c_uint, ctypes.c_char_p]),
        "glLinkProgram": (None, [ctypes.c_uint]),
        "glGetProgramiv": (None, [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]),
        "glUseProgram": (None, [ctypes.c_uint]),
        "glGetUniformLocation": (ctypes.c_int, [ctypes.c_uint, ctypes.c_char_p]),
        "glUniformMatrix4fv": (None, [ctypes.c_int, ctypes.c_int, ctypes.c_ubyte, ctypes.c_void_p]),
        "glUniform1i": (None, [ctypes.c_int, ctypes.c_int]),
        "glUniform3f": (None, [ctypes.c_int] + [ctypes.c_float] * 3),
        "glVertexAttribPointer": (None, [ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_ubyte, ctypes.c_int,
                                         ctypes.c_void_p]),
        "glEnableVertexAttribArray": (None, [ctypes.c_uint]),
        "glDisableVertexAttribArray": (None, [ctypes.c_uint]),
    }
    VERTEX = b"""#version 120
uniform mat4 bones[%d];
uniform mat4 rigid;
uniform int skinned;
uniform int heat;
uniform int sel;
uniform int lit;
uniform vec3 light;
attribute vec4 joints;
attribute vec4 weights;
attribute vec3 normal;
vec3 ramp(float w) {
    vec3 cold = vec3(0.08, 0.12, 0.45), mid = vec3(0.1, 0.85, 0.2), hot = vec3(1.0, 0.15, 0.1);
    return w < 0.5 ? mix(cold, mid, w * 2.0) : mix(mid, hot, (w - 0.5) * 2.0);
}
void main() {
    mat4 m = rigid;
    if (skinned != 0) {
        m = bones[int(joints.x)] * weights.x + bones[int(joints.y)] * weights.y
          + bones[int(joints.z)] * weights.z + bones[int(joints.w)] * weights.w;
    }
    vec4 s = m * vec4(gl_Vertex.xyz, 1.0);
    gl_Position = gl_ModelViewProjectionMatrix * vec4(s.xyz, 1.0);
    float l = 1.0;
    if (lit != 0) {
        vec3 n = (m * vec4(normal, 0.0)).xyz;
        float len = length(n);
        l = len > 0.0001 ? 0.35 + 0.65 * max(0.0, dot(n / len, light)) : 1.0;
    }
    vec4 c = gl_Color;
    if (heat != 0) {
        float w = 0.0;
        if (int(joints.x) == sel) w += weights.x;
        if (int(joints.y) == sel) w += weights.y;
        if (int(joints.z) == sel) w += weights.z;
        if (int(joints.w) == sel) w += weights.w;
        c = vec4(ramp(clamp(w, 0.0, 1.0)), 1.0);
    }
    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_FrontColor = vec4(c.rgb * l, c.a);
}
""" % MAX_BONES
    FRAGMENT = b"""#version 120
uniform sampler2D tex;
uniform int textured;
void main() {
    vec4 c = textured != 0 ? texture2D(tex, gl_TexCoord[0].st) * gl_Color : gl_Color;
    if (c.a < 0.35) discard;
    gl_FragColor = vec4(c.rgb, 1.0);
}
"""

    def __init__(self, lib):
        get = lib.wglGetProcAddress
        get.argtypes, get.restype = [ctypes.c_char_p], ctypes.c_void_p
        for name, (res, args) in self._sigs.items():
            addr = get(name.encode())
            if not addr:
                raise OSError(f"{name} missing")
            setattr(self, name[2:], ctypes.WINFUNCTYPE(res, *args)(addr))
        self.program = self.CreateProgram()
        for kind, src in ((0x8B31, self.VERTEX), (0x8B30, self.FRAGMENT)):
            sh = self.CreateShader(kind)
            self.ShaderSource(sh, 1, ctypes.byref(ctypes.c_char_p(src)), None)
            self.CompileShader(sh)
            ok = ctypes.c_int()
            self.GetShaderiv(sh, 0x8B81, ctypes.byref(ok))
            if not ok.value:
                log = ctypes.create_string_buffer(2048)
                self.GetShaderInfoLog(sh, 2048, None, log)
                self.DeleteShader(sh)
                self.DeleteProgram(self.program)
                raise OSError(log.value.decode("latin-1", "replace"))
            self.AttachShader(self.program, sh)
            self.DeleteShader(sh)
        # Keep generic attributes away from compatibility colour (3), normal (2)
        # and texture-coordinate (8) arrays.
        self.BindAttribLocation(self.program, 6, b"joints")
        self.BindAttribLocation(self.program, 7, b"weights")
        self.BindAttribLocation(self.program, 5, b"normal")
        self.LinkProgram(self.program)
        ok = ctypes.c_int()
        self.GetProgramiv(self.program, 0x8B82, ctypes.byref(ok))
        if not ok.value:
            self.DeleteProgram(self.program)
            raise OSError("shader link")
        self.loc = {n: self.GetUniformLocation(self.program, n.encode())
                    for n in ("bones", "rigid", "skinned", "heat", "sel", "lit", "light", "tex", "textured")}


class GpuMesh:
    """A mesh's arrays for OpenGL: positions (rewritten when blend shapes move), normals, texcoords, colours, the
    four heaviest bones and weights, triangles."""

    def __init__(self, m: igb_model.Mesh, place: list = IDENTITY):
        """place: what puts the vertices in the model (a part that moves with a bone is in that bone's space)."""
        self.mesh = m
        n = len(m.pos)
        pos = m.pos if place is IDENTITY else [igb_model.point(p, place) for p in m.pos]
        nrm = m.nrm if place is IDENTITY or not m.nrm else [igb_model.direction(v, place) for v in m.nrm]
        self.base = array.array("f", [v for p in pos for v in p])
        self.posa = array.array("f", self.base)
        nrm = array.array("f", [v for p in (nrm or [(0.0, 0.0, 0.0)] * n) for v in p])
        uv = array.array("f", [v for t in (m.uv or [(0.0, 0.0)] * n) for v in t])
        neutral = 255 if m.color_scale == 1.0 else 128
        col = array.array("f", [min(1.0, c * m.color_scale / 255.0)
                              for cc in (m.col or [(neutral,) * 4] * n) for c in cc])
        joints, weights = array.array("f"), array.array("f")
        for i in range(n):
            vw = m.weights[i] if m.weights is not None else [(max(0, m.bone), 1.0)]
            top = sorted(vw, key=lambda x: -x[1])[:4]
            total = sum(w for _, w in top) or 1.0
            top += [(0, 0.0)] * (4 - len(top))
            joints.extend(float(min(j, MAX_BONES - 1)) for j, _ in top)
            weights.extend(w / total for _, w in top)
        idx = array.array("I", [i for t in m.tris for i in t])
        self.count = len(idx)
        self.pos = (ctypes.c_float * max(1, 3 * n)).from_buffer(self.posa)
        self.nrm = (ctypes.c_float * max(1, 3 * n)).from_buffer(nrm)
        self.uv = (ctypes.c_float * max(1, 2 * n)).from_buffer(uv)
        self.col = (ctypes.c_float * max(1, 4 * n)).from_buffer(col)
        self.joints = (ctypes.c_float * max(1, 4 * n)).from_buffer(joints)
        self.weights = (ctypes.c_float * max(1, 4 * n)).from_buffer(weights)
        self.idx = (ctypes.c_uint * max(1, self.count)).from_buffer(idx)
        self._keep = (nrm, uv, col, joints, weights, idx)
        self.shape_weights: tuple = ()

    def blend(self, weights: list[float]) -> None:
        """The blend shapes' deltas added at their weights."""
        w = tuple(weights)
        if w == self.shape_weights:
            return
        self.shape_weights = w
        pos = self.posa
        pos[:] = self.base
        for k, (_, deltas) in enumerate(self.mesh.shapes):  # (a skin's: in the model's space as its vertices)
            f = w[k] if k < len(w) else 0.0
            if f:
                for i, d in enumerate(deltas):
                    if d[0] or d[1] or d[2]:
                        pos[3 * i] += d[0] * f
                        pos[3 * i + 1] += d[1] * f
                        pos[3 * i + 2] += d[2] * f


class View(ttk.Frame):
    """The 3D view: an orbit camera round the model (Z up), the meshes, the bones on top."""

    def __init__(self, parent, app: "Viewer"):
        super().__init__(parent)
        self.app = app
        self.canvas = tk.Frame(self, width=640, height=480, background="#1e1f24", takefocus=1)
        self.canvas.pack(fill="both", expand=True)
        self.image_panel = tk.Label(self, background="#272933", borderwidth=0)
        self.preview_photo = None
        self.gl: GL | None = None
        self.sh: Shader | None = None
        self.hdc = self.hglrc = None
        self.gdi32 = None
        self.user32 = None
        self.gpu: dict[int, GpuMesh] = {}
        self.textures: dict[int, int] = {}
        self.target = [0.0, 0.0, 0.5]
        self.yaw, self.pitch, self.dist = 1.25, 0.25, 3.0  # from the front (the characters look down +Y)
        self.drag: tuple | None = None
        self.moved = False
        self.cam = None
        self.size = (1, 1)
        self.offset = [0.0, 0.0, 0.0]  # how far the pose shown has moved the model from its rest (followed)
        self.message = ""
        c = self.canvas
        c.bind("<Map>", lambda e: self.after_idle(self.start_gl))
        c.bind("<Configure>", lambda e: self.draw())
        c.bind("<ButtonPress-1>", lambda e: self.press(e, "turn"))
        c.bind("<ButtonPress-3>", lambda e: self.press(e, "pan"))
        c.bind("<ButtonPress-2>", lambda e: self.press(e, "pan"))
        c.bind("<B1-Motion>", self.motion)
        c.bind("<B3-Motion>", self.motion)
        c.bind("<B2-Motion>", self.motion)
        c.bind("<ButtonRelease-1>", self.release)
        c.bind("<ButtonRelease-3>", lambda e: setattr(self, "drag", None))
        c.bind("<ButtonRelease-2>", lambda e: setattr(self, "drag", None))
        c.bind("<MouseWheel>", self.wheel)
        c.bind("<Double-Button-1>", lambda e: self.frame_model())
        c.bind("<KeyPress-f>", lambda e: self.frame_model())

    # OpenGL.

    def start_gl(self) -> None:
        if self.hglrc:
            return
        if sys.platform != "win32":
            self.app.status('The 3D view requires Windows OpenGL, which is unavailable on this system.')
            return
        try:
            self.gl = GL()
            self.gl.lib.glPointSize.argtypes, self.gl.lib.glPointSize.restype = [ctypes.c_float], None
            self.gl.PointSize = self.gl.lib.glPointSize
            user32, gdi32 = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32")
            user32.GetDC.argtypes, user32.GetDC.restype = [ctypes.c_void_p], ctypes.c_void_p
            gdi32.ChoosePixelFormat.argtypes = [ctypes.c_void_p, ctypes.POINTER(PIXELFORMATDESCRIPTOR)]
            gdi32.SetPixelFormat.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(PIXELFORMATDESCRIPTOR)]
            gdi32.SwapBuffers.argtypes = [ctypes.c_void_p]
            self.gdi32 = gdi32
            self.user32 = user32
            user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.hdc = user32.GetDC(self.canvas.winfo_id())
            pfd = PIXELFORMATDESCRIPTOR(nSize=ctypes.sizeof(PIXELFORMATDESCRIPTOR), nVersion=1,
                                        dwFlags=0x4 | 0x20 | 0x1, iPixelType=0, cColorBits=32, cDepthBits=24,
                                        cAlphaBits=8)
            fmt = gdi32.ChoosePixelFormat(self.hdc, ctypes.byref(pfd))
            if not fmt or not gdi32.SetPixelFormat(self.hdc, fmt, ctypes.byref(pfd)):
                raise OSError("pixel format")
            self.hglrc = self.gl.wglCreateContext(self.hdc)
            if not self.hglrc or not self.gl.wglMakeCurrent(self.hdc, self.hglrc):
                raise OSError("OpenGL context")
            self.sh = Shader(self.gl.lib)
        except (OSError, AttributeError) as e:
            if self.gl and self.hglrc:
                self.gl.wglMakeCurrent(None, None)
                self.gl.wglDeleteContext(self.hglrc)
            if self.hdc and self.user32:
                self.user32.ReleaseDC(self.canvas.winfo_id(), self.hdc)
                self.hdc = None
            self.gl = self.sh = None
            self.hglrc = None
            self.app.status(tr('OpenGL unavailable ({0}): the 3D view is disabled; import and export remain available.', e))
            return
        self.draw()

    def current(self) -> bool:
        return bool(self.gl and self.hglrc and self.gl.wglMakeCurrent(self.hdc, self.hglrc))

    def reset(self) -> None:
        """A new model: its arrays and textures made again when drawn."""
        if self.textures and self.current():
            ids = (ctypes.c_uint * len(self.textures))(*self.textures.values())
            self.gl.DeleteTextures(len(self.textures), ids)
        self.textures = {}
        self.gpu = {}
        self.image_panel.pack_forget()
        self.image_panel.configure(image="")
        self.preview_photo = None
        self.canvas.pack(fill="both", expand=True)

    def show_image(self, texture) -> None:
        """Texture-only IGBs, especially white font glyphs, need an image view."""
        side = max(128, min(self.winfo_width(), self.winfo_height(), 900))
        self.preview_photo = photo(texture.w, texture.h, texture.rgba, side)
        self.image_panel.configure(image=self.preview_photo)
        self.canvas.pack_forget()
        self.image_panel.pack(fill="both", expand=True)

    def texture(self, k: int) -> int:
        model = self.app.model
        if k < 0 or model is None or k >= len(model.textures):
            return 0
        if k not in self.textures:
            t = model.textures[k]
            tex = ctypes.c_uint()
            gl = self.gl
            gl.GenTextures(1, ctypes.byref(tex))
            gl.BindTexture(GL_TEXTURE_2D, tex.value)
            sampling = GL_NEAREST if self.app.filter_var_3d.get() == "Pixel" else GL_LINEAR
            for p, v in ((GL_TEXTURE_MIN_FILTER, sampling), (GL_TEXTURE_MAG_FILTER, sampling),
                         (GL_TEXTURE_WRAP_S, GL_REPEAT), (GL_TEXTURE_WRAP_T, GL_REPEAT)):
                gl.TexParameteri(GL_TEXTURE_2D, p, v)
            gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1)
            gl.TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t.w, t.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, t.rgba)
            self.textures[k] = tex.value
        return self.textures[k]

    def filter_changed(self) -> None:
        if self.current():
            sampling = GL_NEAREST if self.app.filter_var_3d.get() == "Pixel" else GL_LINEAR
            for tex in self.textures.values():
                self.gl.BindTexture(GL_TEXTURE_2D, tex)
                for p in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
                    self.gl.TexParameteri(GL_TEXTURE_2D, p, sampling)
        self.draw()

    def drop_texture(self, k: int) -> None:
        if k in self.textures and self.current():
            self.gl.DeleteTextures(1, (ctypes.c_uint * 1)(self.textures.pop(k)))

    # The camera.

    def frame_model(self) -> None:
        model = self.app.model
        if model is None:
            return
        shown = [m for m in model.meshes if self.app.visible(m)]
        lo, hi = bounds_of(model, shown)
        self.offset = self.pose_offset()
        self.target = [(a + b) / 2 + o for a, b, o in zip(lo, hi, self.offset)]
        size = max(1e-3, max(b - a for a, b in zip(lo, hi)))
        self.dist = size * 1.6
        self.draw()

    def pose_offset(self) -> list[float]:
        """Where the pose shown has taken the model's main bone (the one most of the skin hangs on) from its rest:
        the diorama's animations put Tiny where he stands in the diorama, far from where he was modelled."""
        model = self.app.model
        if model is None or not model.bones:
            return [0.0, 0.0, 0.0]
        count = [0] * len(model.bones)
        for m in model.meshes:
            for vw in m.weights or ():
                for b, w in vw:
                    if w > 0.5 and b < len(count):
                        count[b] += 1
        ref = max(range(len(count)), key=count.__getitem__) if any(count) else 0
        pose = self.app.pose()
        if not pose or ref >= len(pose):
            return [0.0, 0.0, 0.0]
        return [pose[ref][12 + k] - model.bones[ref].bind[12 + k] for k in range(3)]

    def follow(self) -> None:
        """The camera moved with the model when another pose takes it elsewhere."""
        off = self.pose_offset()
        self.target = [t + n - o for t, n, o in zip(self.target, off, self.offset)]
        self.offset = off
        self.draw()

    def eye(self) -> list[float]:
        cp = math.cos(self.pitch)
        return [self.target[0] + self.dist * cp * math.cos(self.yaw),
                self.target[1] + self.dist * cp * math.sin(self.yaw),
                self.target[2] + self.dist * math.sin(self.pitch)]

    def basis(self) -> tuple[list, list, list, list]:
        e = self.eye()
        f = [t - x for t, x in zip(self.target, e)]
        n = math.sqrt(sum(v * v for v in f)) or 1.0
        f = [v / n for v in f]
        r = [f[1], -f[0], 0.0]  # f x up (0, 0, 1)
        n = math.sqrt(sum(v * v for v in r)) or 1.0
        r = [v / n for v in r] if n > 1e-6 else [1.0, 0.0, 0.0]
        u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
        return e, f, r, u

    def press(self, e, mode: str) -> None:
        self.canvas.focus_set()
        self.drag = (mode, e.x, e.y, self.yaw, self.pitch, list(self.target))
        self.moved = False

    def motion(self, e) -> None:
        if not self.drag:
            return
        mode, x0, y0, yaw, pitch, target = self.drag
        dx, dy = e.x - x0, e.y - y0
        if abs(dx) + abs(dy) > 3:
            self.moved = True
        if mode == "turn":
            self.yaw = yaw - dx * 0.01
            self.pitch = max(-1.55, min(1.55, pitch + dy * 0.01))
        else:
            _, _, r, u = self.basis()
            k = self.dist * 0.0015
            self.target = [t - r[i] * dx * k + u[i] * dy * k for i, t in enumerate(target)]
        self.draw()

    def release(self, e) -> None:
        if self.drag and not self.moved:
            self.pick(e.x, e.y)
        self.drag = None

    def wheel(self, e) -> None:
        self.dist *= 0.88 if e.delta > 0 else 1 / 0.88
        self.draw()

    def project(self, p) -> tuple[float, float, float] | None:
        """A point on the screen (x, y in pixels, its depth), None behind the camera."""
        e, f, r, u = self.basis()
        d = [p[k] - e[k] for k in range(3)]
        z = sum(d[k] * f[k] for k in range(3))
        if z < 1e-4:
            return None
        w, h = self.size
        top = math.tan(FOV / 2)
        x = sum(d[k] * r[k] for k in range(3)) / z / (top * w / h)
        y = sum(d[k] * u[k] for k in range(3)) / z / top
        return (x + 1) / 2 * w, (1 - y) / 2 * h, z

    def pick(self, x: int, y: int) -> None:
        """The bone whose joint is nearest the click (within 12 pixels)."""
        model = self.app.model
        if model is None or not model.bones or not self.app.bones_var.get():
            return
        pose = self.app.pose()
        best, best_d = None, 144.0
        for i in range(len(model.bones)):
            s = self.project(pose[i][12:15])
            if s is None:
                continue
            d = (s[0] - x) ** 2 + (s[1] - y) ** 2
            if d < best_d:
                best, best_d = i, d
        if best is not None:
            self.app.select_bone(best)

    # Drawing.

    def draw(self, size: tuple[int, int] | None = None) -> None:
        if self.preview_photo is not None and size is None:
            return
        if not self.current():
            return
        gl, app = self.gl, self.app
        w, h = size or (max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        self.size = (w, h)
        gl.Viewport(0, 0, w, h)
        gl.ClearColor(0.12, 0.125, 0.145, 1.0)
        gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        model = app.model
        near = max(1e-3, self.dist * 0.01)
        gl.MatrixMode(GL_PROJECTION)
        gl.LoadIdentity()
        top = near * math.tan(FOV / 2)
        gl.Frustum(-top * w / h, top * w / h, -top, top, near, near * 20000)
        gl.MatrixMode(GL_MODELVIEW)
        e, f, r, u = self.basis()
        view = [r[0], u[0], -f[0], 0.0, r[1], u[1], -f[1], 0.0, r[2], u[2], -f[2], 0.0,
                -sum(r[k] * e[k] for k in range(3)), -sum(u[k] * e[k] for k in range(3)),
                sum(f[k] * e[k] for k in range(3)), 1.0]
        gl.LoadMatrixf((ctypes.c_float * 16)(*view))
        gl.Enable(GL_DEPTH_TEST)
        gl.DepthFunc(GL_LEQUAL)
        gl.Disable(GL_CULL_FACE)
        gl.Disable(GL_BLEND)
        if app.grid_var.get():
            self.draw_grid(model)
        if model is not None:
            pose = app.pose()
            if app.mesh_var.get():
                self.draw_meshes(model, pose, f)
            if app.bones_var.get() and model.bones:
                self.draw_bones(model, pose)
            if model.scene_motion and not model.bones and not model.meshes:
                self.draw_scene_motion(model)
        if size is None:
            self.gdi32.SwapBuffers(self.hdc)

    def draw_grid(self, model) -> None:
        gl = self.gl
        lo, hi = bounds_of(model, [m for m in model.meshes if self.app.visible(m)]) if model else \
            ((-1, -1, 0), (1, 1, 1))
        size = max(hi[0] - lo[0], hi[1] - lo[1], 1e-3)
        step = 10 ** math.floor(math.log10(size / 2))
        n = int(size / step) + 2
        cx, cy = round(self.target[0] / step) * step, round(self.target[1] / step) * step
        z = lo[2] + self.offset[2] if model and model.meshes else 0.0
        gl.Begin(GL_LINES)
        for k in range(-n, n + 1):
            c = 0.32 if k else 0.45
            gl.Color4f(c, c, c + 0.03, 1.0)
            gl.Vertex3f(cx + k * step, cy - n * step, z)
            gl.Vertex3f(cx + k * step, cy + n * step, z)
            gl.Vertex3f(cx - n * step, cy + k * step, z)
            gl.Vertex3f(cx + n * step, cy + k * step, z)
        gl.End()

    def draw_meshes(self, model: igb_model.Model, pose: list | None, fwd: list) -> None:
        gl, sh, app = self.gl, self.sh, self.app
        if sh is None:
            return
        skins = model.skin_matrices(pose)[:MAX_BONES]
        palette = (ctypes.c_float * (16 * max(1, len(skins))))(*[v for m in skins or [IDENTITY] for v in m])
        sh.UseProgram(sh.program)
        sh.UniformMatrix4fv(sh.loc["bones"], max(1, len(skins)), 0, palette)
        sh.Uniform1i(sh.loc["lit"], 1 if app.light_var.get() else 0)
        sh.Uniform3f(sh.loc["light"], *[-v for v in fwd])
        sh.Uniform1i(sh.loc["tex"], 0)
        sel = app.bone if app.bone is not None else -1
        heat = 1 if app.heat_var.get() and sel >= 0 else 0
        gl.PolygonMode(GL_FRONT_AND_BACK, GL_LINE if app.wire_var.get() else GL_FILL)
        gl.EnableClientState(GL_VERTEX_ARRAY)
        gl.EnableClientState(GL_COLOR_ARRAY)
        for a in (6, 7, 5):
            sh.EnableVertexAttribArray(a)
        for k, m in enumerate(model.meshes):
            if not app.visible(m) or not m.tris:
                continue
            g = self.gpu.get(k)
            if g is None or g.mesh is not m:
                on_bone = not m.skinned and 0 <= m.bone < len(model.bones)
                g = self.gpu[k] = GpuMesh(m, model.bones[m.bone].bind if on_bone else IDENTITY)
            if m.shapes:
                g.blend(app.shape_weights(k))
            skinned = m.skinned or 0 <= m.bone < len(model.bones)
            # (more bones than the shader takes: drawn in the rest pose)
            sh.Uniform1i(sh.loc["skinned"], 1 if skinned and len(model.bones) <= MAX_BONES else 0)
            clip = model.clips[app.clip] if app.clip is not None and app.clip < len(model.clips) else None
            rigid = model.mesh_matrix(m, clip, app.time) if not skinned else IDENTITY
            sh.UniformMatrix4fv(sh.loc["rigid"], 1, 0, (ctypes.c_float * 16)(*rigid))
            sh.Uniform1i(sh.loc["heat"], heat if m.skinned or m.bone >= 0 else 0)
            sh.Uniform1i(sh.loc["sel"], sel)
            tex = self.texture(m.texture) if app.tex_var.get() and m.uv and not (heat and skinned) else 0
            if tex:
                gl.Enable(GL_TEXTURE_2D)
                gl.BindTexture(GL_TEXTURE_2D, tex)
                gl.EnableClientState(GL_TEXTURE_COORD_ARRAY)
                gl.TexCoordPointer(2, GL_FLOAT, 0, g.uv)
            else:
                gl.Disable(GL_TEXTURE_2D)
                gl.DisableClientState(GL_TEXTURE_COORD_ARRAY)
            sh.Uniform1i(sh.loc["textured"], 1 if tex else 0)
            gl.VertexPointer(3, GL_FLOAT, 0, g.pos)
            gl.ColorPointer(4, GL_FLOAT, 0, g.col)
            sh.VertexAttribPointer(6, 4, GL_FLOAT, 0, 0, g.joints)
            sh.VertexAttribPointer(7, 4, GL_FLOAT, 0, 0, g.weights)
            sh.VertexAttribPointer(5, 3, GL_FLOAT, 0, 0, g.nrm)
            gl.DrawElements(GL_TRIANGLES, g.count, GL_UNSIGNED_INT, g.idx)
        for a in (6, 7, 5):
            sh.DisableVertexAttribArray(a)
        gl.DisableClientState(GL_TEXTURE_COORD_ARRAY)
        gl.DisableClientState(GL_COLOR_ARRAY)
        gl.DisableClientState(GL_VERTEX_ARRAY)
        gl.Disable(GL_TEXTURE_2D)
        gl.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        sh.UseProgram(0)

    def draw_scene_motion(self, model) -> None:
        """A standalone camera/helper file has a moving transform, rather than a mesh."""
        motion, gl = model.scene_motion, self.gl
        clip = model.clips[self.app.clip] if self.app.clip is not None else None
        phase = getattr(clip, "scene_phase", 0)
        for index in motion.nodes:
            if (index, phase) not in motion.channels:
                continue
            gl.Color4f(.3, .7, 1.0, 1.0)
            gl.Begin(3)  # GL_LINE_STRIP
            for p in motion.path(index, phase):
                gl.Vertex3f(*p)
            gl.End()
            matrix = motion.local(index, phase, self.app.time)
            p = matrix[12:15]
            gl.PointSize(10)
            gl.Begin(GL_POINTS)
            gl.Color4f(1, .8, .1, 1)
            gl.Vertex3f(*p)
            gl.End()
            gl.PointSize(1)
            length = max(.05, self.dist * .035)
            gl.Begin(GL_LINES)
            for axis, color in enumerate(((1, .2, .2), (.2, 1, .2), (.2, .4, 1))):
                gl.Color4f(*color, 1)
                gl.Vertex3f(*p)
                v = igb_model.normalised(matrix[axis * 4:axis * 4 + 3])
                gl.Vertex3f(*[p[k] + length * v[k] for k in range(3)])
            gl.End()

    def draw_bones(self, model: igb_model.Model, pose: list) -> None:
        gl, app = self.gl, self.app
        sel = app.bone
        far = self.dist * 50
        pts = [p[12:15] for p in pose]
        ok = [all(abs(v - t) < far for v, t in zip(p, self.target)) for p in pts]
        gl.Disable(GL_DEPTH_TEST)
        gl.LineWidth(2.0)
        gl.Begin(GL_LINES)
        for i, b in enumerate(model.bones):
            p = b.parent
            if p < 0 or not ok[i] or not ok[p]:
                continue
            if i == sel:
                gl.Color4f(1.0, 0.85, 0.1, 1.0)
            elif p == sel:
                gl.Color4f(1.0, 0.55, 0.15, 1.0)
            else:
                gl.Color4f(0.25, 0.85, 1.0, 1.0)
            gl.Vertex3f(*pts[p])
            gl.Vertex3f(*pts[i])
        gl.End()
        gl.PointSize(6.0)
        gl.Begin(GL_POINTS)
        for i in range(len(model.bones)):
            if ok[i] and i != sel:
                gl.Color4f(0.9, 0.95, 1.0, 1.0)
                gl.Vertex3f(*pts[i])
        gl.End()
        if sel is not None and 0 <= sel < len(pts) and ok[sel]:
            gl.PointSize(11.0)
            gl.Begin(GL_POINTS)
            gl.Color4f(1.0, 0.85, 0.1, 1.0)
            gl.Vertex3f(*pts[sel])
            gl.End()
            # Its axes (x red, y green, z blue).
            m = pose[sel]
            k = self.dist * 0.06
            gl.Begin(GL_LINES)
            for axis, col in ((0, (1.0, 0.25, 0.25)), (1, (0.3, 1.0, 0.3)), (2, (0.35, 0.5, 1.0))):
                d = m[axis * 4:axis * 4 + 3]
                n = math.sqrt(sum(v * v for v in d)) or 1.0
                gl.Color4f(*col, 1.0)
                gl.Vertex3f(*pts[sel])
                gl.Vertex3f(*[pts[sel][j] + d[j] / n * k for j in range(3)])
            gl.End()
        gl.PointSize(1.0)
        gl.LineWidth(1.0)
        gl.Enable(GL_DEPTH_TEST)

    def snapshot(self, w: int, h: int) -> tuple[int, int, bytes] | None:
        """The view drawn into a framebuffer of w x h (not on screen): RGBA rows from the top."""
        if not self.current():
            return None
        gl = self.gl
        get = gl.lib.wglGetProcAddress
        get.argtypes, get.restype = [ctypes.c_char_p], ctypes.c_void_p
        u, vp = ctypes.c_uint, ctypes.c_void_p

        def fn(name, *args):
            addr = get(name.encode())
            if not addr:
                raise OSError(f"{name} missing")
            return ctypes.WINFUNCTYPE(None, *args)(addr)
        try:
            gen_fb, bind_fb = fn("glGenFramebuffers", ctypes.c_int, vp), fn("glBindFramebuffer", u, u)
            gen_rb, bind_rb = fn("glGenRenderbuffers", ctypes.c_int, vp), fn("glBindRenderbuffer", u, u)
            storage = fn("glRenderbufferStorage", u, u, ctypes.c_int, ctypes.c_int)
            attach = fn("glFramebufferRenderbuffer", u, u, u, u)
            del_fb, del_rb = fn("glDeleteFramebuffers", ctypes.c_int, vp), fn("glDeleteRenderbuffers", ctypes.c_int, vp)
        except OSError:
            return None
        fb, rb = ctypes.c_uint(), (ctypes.c_uint * 2)()
        gen_fb(1, ctypes.byref(fb))
        gen_rb(2, rb)
        bind_fb(0x8D40, fb.value)
        bind_rb(0x8D41, rb[0])
        storage(0x8D41, 0x8058, w, h)
        attach(0x8D40, 0x8CE0, 0x8D41, rb[0])
        bind_rb(0x8D41, rb[1])
        storage(0x8D41, 0x81A6, w, h)
        attach(0x8D40, 0x8D00, 0x8D41, rb[1])
        buf = (ctypes.c_ubyte * (w * h * 4))()
        keep = self.size
        try:
            self.draw((w, h))
            gl.PixelStorei(GL_PACK_ALIGNMENT, 1)
            gl.ReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf)
        finally:
            bind_fb(0x8D40, 0)
            del_rb(2, rb)
            del_fb(1, ctypes.byref(fb))
            self.size = keep
        data = bytearray(b"".join(bytes(buf[(h - 1 - y) * w * 4:(h - y) * w * 4]) for y in range(h)))
        data[3::4] = b"\xff" * (w * h)
        return w, h, bytes(data)

    def destroy(self) -> None:
        self.reset()
        if self.gl and self.hglrc:
            if self.sh and self.current():
                self.sh.DeleteProgram(self.sh.program)
            self.gl.wglMakeCurrent(None, None)
            self.gl.wglDeleteContext(self.hglrc)
            self.hglrc = None
        if self.hdc and self.user32:
            self.user32.ReleaseDC(self.canvas.winfo_id(), self.hdc)
            self.hdc = None
        super().destroy()


def bounds_of(model: igb_model.Model | None, meshes: list) -> tuple[tuple, tuple]:
    """The box round the meshes: round the biggest one and those near it (a prop the rest pose leaves far away,
    like the diorama's dumbbell, left out)."""
    if model is None:
        return (-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)
    ground = [m for m in meshes if "sky" not in m.name.lower()]  # a track's sky dome round everything
    meshes = ground or meshes
    if len(meshes) > 12:  # a track or a level: all of it
        pts = []
        for m in meshes:
            place = model.bones[m.bone].bind if 0 <= m.bone < len(model.bones) else \
                (IDENTITY if m.skinned else m.matrix)
            pts += [igb_model.point(p, place) for p in m.pos[::max(1, len(m.pos) // 50)]]
        return tuple(min(p[k] for p in pts) for k in range(3)), tuple(max(p[k] for p in pts) for k in range(3))
    boxes = []
    for m in sorted(meshes, key=lambda m: -len(m.pos)):
        place = model.bones[m.bone].bind if 0 <= m.bone < len(model.bones) else \
            (IDENTITY if m.skinned else m.matrix)
        ps = [igb_model.point(p, place) for p in m.pos[::max(1, len(m.pos) // 500)]]
        if ps:
            boxes.append(ps)
    pts = []
    if boxes:
        main = boxes[0]
        lo = [min(p[k] for p in main) for k in range(3)]
        hi = [max(p[k] for p in main) for k in range(3)]
        mid = [(a + b) / 2 for a, b in zip(lo, hi)]
        size = max(max(b - a for a, b in zip(lo, hi)), 1e-3)
        for ps in boxes:
            c = [sum(p[k] for p in ps) / len(ps) for k in range(3)]
            if ps is main or max(abs(c[k] - mid[k]) for k in range(3)) < 2 * size:
                pts += ps
    if not pts:
        pts = [tuple(b.bind[12:15]) for b in model.bones if max(abs(v) for v in b.bind[12:15]) < 100]
    if not pts and model.scene_motion:
        motion = model.scene_motion
        pts = [p for index, phase in motion.channels for p in motion.path(index, phase)
               if all(math.isfinite(v) for v in p)]
    if not pts:
        return (-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)
    return tuple(min(p[k] for p in pts) for k in range(3)), tuple(max(p[k] for p in pts) for k in range(3))


def raw_insert(box: tk.Listbox, text: str) -> None:
    """A line into a list as it is (file and bone names are not to be translated)."""
    box.tk.call(box._w, "insert", "end", text)


def raw_label(label, text: str) -> None:
    """A label's text as it is (already translated, with names in it)."""
    label.tk.call(label._w, "configure", "-text", text)


def raw_text(tree: ttk.Treeview, iid: str, text: str) -> None:
    tree.tk.call(tree._w, "item", iid, "-text", text)


def photo(w: int, h: int, rgba: bytes, side: int) -> tk.PhotoImage:
    """A picture for Tk, at most side x side (shrunk by whole steps, or blown up by whole steps)."""
    ph = tk.PhotoImage(data=base64.b64encode(png_util.png_bytes(w, h, rgba)).decode("ascii"), format="png")
    big = max(w, h)
    if big > side:
        k = -(-big // side)
        return ph.subsample(k, k)
    if big * 2 <= side:
        k = side // big
        return ph.zoom(k, k)
    return ph


class Viewer(tk.Tk):
    def __init__(self, path: Path | None):
        super().__init__()
        self.title('IGB Viewer — Crash Nitro Kart')
        self.geometry("1280x820")
        self.minsize(980, 640)
        self.settings = load_settings()
        self.base: igb_model.Model | None = None      # the .igb open (what is saved into)
        self.imported: igb_model.Model | None = None  # a glTF imported over it (what is shown and saved)
        self.model: igb_model.Model | None = None     # what is shown
        self.base_data: bytes | None = None
        self.base_name = ""
        self.tex_changes: dict[int, tuple[int, int, bytes]] = {}  # the base's textures replaced by PNGs
        self.gob: cnk_gob.Gob | None = None
        self.gob_entries: list[cnk_gob.Entry] = []
        self.hidden: set[int] = set()
        self.bone: int | None = None
        self.clip: int | None = None
        self.time = 0.0
        self.playing = False
        self.last_tick = time.perf_counter()
        self.shape_vars: dict[int, list[tk.DoubleVar]] = {}
        self.bone_offsets: dict[int, tuple] = {}
        self.face_indices: list[int] = []
        self.updating_face = False
        self.images: dict[str, tk.PhotoImage] = {}
        self.build()
        self.protocol("WM_DELETE_WINDOW", self.quit_viewer)
        gfc = self.settings.get("igb_gfc")
        if gfc and Path(gfc).is_file():
            self.after(30, lambda: self.open_gob(Path(gfc), quiet=True))
        if path:
            self.after(60, lambda: self.open_path(path))
        self.tick()

    # Layout.

    def build(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Head.TLabel", font=("Segoe UI", 11, "bold"))
        bar = ttk.Frame(self, padding=(6, 6, 6, 2))
        bar.pack(fill="x")
        ttk.Button(bar, text='Open .igb…', command=self.ask_open).pack(side="left")
        ttk.Button(bar, text='Open ASSETS.GFC…', command=self.ask_gob).pack(side="left", padx=(4, 0))
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(bar, text='Export glTF…', command=self.export_gltf).pack(side="left")
        ttk.Button(bar, text='Export OBJ…', command=self.export_obj).pack(side="left", padx=(4, 0))
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(bar, text='Import glTF…', command=self.import_gltf).pack(side="left")
        self.undo_button = ttk.Button(bar, text='Discard import', command=self.drop_import, state="disabled")
        self.undo_button.pack(side="left", padx=(4, 0))
        self.save_button = ttk.Button(bar, text='Save .igb…', command=self.save_igb, state="disabled")
        self.save_button.pack(side="left", padx=(4, 0))
        ttk.Button(bar, text="?", width=3, command=self.help).pack(side="right")
        self.language_choice = i18n.selector(bar, self.settings, save_settings, self.reopen)
        self.language_choice.pack(side="right", padx=(0, 6))
        self.title_label = ttk.Label(self, text='No file open', style="Head.TLabel", padding=(8, 2))
        self.title_label.pack(fill="x")
        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=6, pady=4)
        sidebar = ttk.Frame(body, width=380)
        body.add(sidebar, weight=0)
        self.tab_choice = ttk.Combobox(sidebar, state="readonly")
        self.tab_choice.pack(fill="x", pady=(0, 4))
        style.layout("Pages.TNotebook.Tab", [])
        left = ttk.Notebook(sidebar, width=380, style="Pages.TNotebook")
        left.pack(fill="both", expand=True)
        right = ttk.Frame(body)
        body.add(right, weight=1)
        self.tabs = left
        self.build_archive(left)
        self.build_meshes(left)
        self.build_bones(left)
        self.build_anims(left)
        self.build_face(left)
        self.build_shapes(left)
        self.build_textures(left)
        self.inspector = Inspector(left)
        left.add(self.inspector, text='IGB data')
        self.tab_choice.configure(values=[left.tab(t, "text") for t in left.tabs()])
        self.tab_choice.current(0)
        self.tab_choice.bind("<<ComboboxSelected>>", lambda e: left.select(self.tab_choice.current()))
        left.bind("<<NotebookTabChanged>>", lambda e: self.tab_choice.current(left.index("current")))
        opts = ttk.Frame(right)
        opts.pack(fill="x")
        self.mesh_var = tk.BooleanVar(value=True)
        self.tex_var = tk.BooleanVar(value=True)
        self.light_var = tk.BooleanVar(value=True)
        self.wire_var = tk.BooleanVar(value=False)
        self.bones_var = tk.BooleanVar(value=True)
        self.heat_var = tk.BooleanVar(value=True)
        self.grid_var = tk.BooleanVar(value=True)
        for text, var in (("Mesh", self.mesh_var), ("Texture", self.tex_var), ('Lighting', self.light_var),
                          ('Wireframe', self.wire_var), ('Bones', self.bones_var),
                          ('Selected bone weights', self.heat_var), ('Grid', self.grid_var)):
            ttk.Checkbutton(opts, text=text, variable=var, command=self.redraw).pack(side="left", padx=(0, 8))
        ttk.Button(opts, text='Frame model', command=lambda: self.view.frame_model()).pack(side="right")
        self.view = View(right, self)
        self.view.pack(fill="both", expand=True, pady=(4, 0))
        ttk.Label(right, text='Left: orbit (click a joint to select its bone) — right: pan — wheel: zoom — double-click or F: frame model', foreground="#666").pack(fill="x")
        self.status_label = ttk.Label(self, text="", relief="sunken", anchor="w", padding=(6, 2))
        self.status_label.pack(fill="x", side="bottom")

    def build_archive(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb, padding=6)
        nb.add(f, text='Archive')
        self.gob_label = ttk.Label(f, text='Open ASSETS.GFC to browse models, tracks, images and other files.',
                                   wraplength=330, justify="left")
        self.gob_label.pack(fill="x")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(6, 4))
        ttk.Label(row, text='Search:').pack(side="left")
        previous_filter = self.settings.get("igb_filter", "")
        if not self.settings.get("archive_browser_v2") and previous_filter.strip() == "chars":
            previous_filter = ""
        self.filter_var = tk.StringVar(value=previous_filter)
        ent = ttk.Entry(row, textvariable=self.filter_var)
        ent.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.filter_var.trace_add("write", lambda *a: self.fill_gob())
        ttk.Button(row, text='Reset', command=lambda: self.filter_var.set("")).pack(side="left", padx=(4, 0))
        self.archive_category = tk.StringVar(value=tr('All'))
        category = self.category_choice = ttk.Combobox(f, textvariable=self.archive_category,
                                    values=[tr(name) for name in cnk_assets.CATEGORIES], state="readonly")
        category.pack(fill="x", pady=(0, 4))
        category.bind("<<ComboboxSelected>>", lambda e: self.fill_gob())
        self.archive_count = ttk.Label(f)
        self.archive_count.pack(fill="x", pady=(0, 4))
        lf = ttk.Frame(f)
        lf.pack(fill="both", expand=True)
        self.gob_list = tk.Listbox(lf, activestyle="dotbox", exportselection=False)
        sb = ttk.Scrollbar(lf, command=self.gob_list.yview)
        self.gob_list.configure(yscrollcommand=sb.set)
        self.gob_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.gob_list.bind("<Double-Button-1>", lambda e: self.open_from_gob())
        self.gob_list.bind("<Return>", lambda e: self.open_from_gob())
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text='Open', command=self.open_from_gob).pack(side="left")
        ttk.Button(row, text='Extract file…', command=self.extract_from_gob).pack(side="left", padx=(4, 0))

    def build_meshes(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb, padding=6)
        nb.add(f, text="Mesh")
        ttk.Label(f, text='Click a checkbox to show or hide a mesh. glTF and OBJ exports include visible meshes.', wraplength=330, justify="left").pack(fill="x")
        cols = ("vis", "verts", "tris", "kind")
        self.mesh_tree = ttk.Treeview(f, columns=cols, show="tree headings", selectmode="browse", height=12)
        self.mesh_tree.heading("#0", text="Mesh")
        self.mesh_tree.heading("vis", text="")
        self.mesh_tree.heading("verts", text='Vertices')
        self.mesh_tree.heading("tris", text='Triangles')
        self.mesh_tree.heading("kind", text='Type')
        self.mesh_tree.column("#0", width=130)
        self.mesh_tree.column("vis", width=26, anchor="center", stretch=False)
        self.mesh_tree.column("verts", width=55, anchor="e", stretch=False)
        self.mesh_tree.column("tris", width=60, anchor="e", stretch=False)
        self.mesh_tree.column("kind", width=90, stretch=False)
        self.mesh_tree.pack(fill="both", expand=True, pady=(6, 0))
        self.mesh_tree.bind("<ButtonRelease-1>", self.mesh_click)
        self.mesh_info = ttk.Label(f, text="", wraplength=330, justify="left", foreground="#444")
        self.mesh_info.pack(fill="x", pady=(6, 0))
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text='Show all', command=lambda: self.show_meshes(True)).pack(side="left")
        ttk.Button(row, text='Best LOD only', command=lambda: self.show_meshes(False)).pack(side="left",
                                                                                                padx=(4, 0))

    def build_bones(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb, padding=6)
        nb.add(f, text='Bones')
        self.bone_tree = ttk.Treeview(f, columns=("bm",), show="tree headings", selectmode="browse")
        self.bone_tree.heading("#0", text='Bone')
        self.bone_tree.heading("bm", text='Matrix')
        self.bone_tree.column("#0", width=230)
        self.bone_tree.column("bm", width=60, anchor="e", stretch=False)
        sb = ttk.Scrollbar(f, command=self.bone_tree.yview)
        self.bone_tree.configure(yscrollcommand=sb.set)
        lower = ttk.Frame(f)
        lower.pack(side="bottom", fill="x")
        self.bone_info = ttk.Label(lower, text="", wraplength=330, justify="left", foreground="#444")
        self.bone_info.pack(fill="x", pady=(6, 0))
        ttk.Label(lower, text=HEAT_HELP, foreground="#666").pack(fill="x")
        self.bone_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.bone_tree.bind("<<TreeviewSelect>>", self.bone_picked)

    def build_anims(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb, padding=6)
        nb.add(f, text='Animations')
        lf = ttk.Frame(f)
        lf.pack(fill="both", expand=True)
        self.anim_list = tk.Listbox(lf, activestyle="dotbox", exportselection=False)
        sb = ttk.Scrollbar(lf, command=self.anim_list.yview)
        self.anim_list.configure(yscrollcommand=sb.set)
        self.anim_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.anim_list.bind("<<ListboxSelect>>", lambda e: self.anim_picked())
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(6, 0))
        self.play_button = ttk.Button(row, text="▶ Play", width=9, command=self.toggle_play)
        self.play_button.pack(side="left")
        ttk.Button(row, text='Rest pose', command=self.rest_pose).pack(side="left", padx=(4, 0))
        ttk.Label(row, text='Speed:').pack(side="left", padx=(10, 2))
        self.speed_var = tk.StringVar(value="1×")
        ttk.Combobox(row, textvariable=self.speed_var, values=("0.25×", "0.5×", "1×", "2×"), width=6,
                     state="readonly").pack(side="left")
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_scale = ttk.Scale(f, from_=0.0, to=1.0, variable=self.time_var, command=self.scrub)
        self.time_scale.pack(fill="x", pady=(6, 0))
        self.time_label = ttk.Label(f, text='rest pose')
        self.time_label.pack(fill="x")

    def build_shapes(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb, padding=6)
        nb.add(f, text="Blendshape")
        top = ttk.Frame(f)
        top.pack(fill="x")
        ttk.Button(top, text='Reset', command=self.zero_shapes).pack(side="left")
        holder = ttk.Frame(f)
        holder.pack(fill="both", expand=True, pady=(6, 0))
        self.shape_canvas = tk.Canvas(holder, highlightthickness=0, background="#f5f5f5")
        sb = ttk.Scrollbar(holder, command=self.shape_canvas.yview)
        self.shape_canvas.configure(yscrollcommand=sb.set)
        self.shape_canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.shape_frame = ttk.Frame(self.shape_canvas)
        self.shape_canvas.create_window((0, 0), window=self.shape_frame, anchor="nw")
        self.shape_frame.bind("<Configure>", lambda e: self.shape_canvas.configure(
            scrollregion=self.shape_canvas.bbox("all")))

    def build_face(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb, padding=6)
        nb.add(f, text='Face / pose')
        ttk.Label(f, text="Expressions use the model's actual bones. Edits preview on top of the animation and do not change the original file.",
                  wraplength=350, justify="left").pack(fill="x", pady=(0, 6))
        self.face_only = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text='Facial bones only', variable=self.face_only,
                        command=self.fill_face).pack(anchor="w")
        self.face_choice = ttk.Combobox(f, state="readonly")
        self.face_choice.pack(fill="x", pady=6)
        self.face_choice.bind("<<ComboboxSelected>>", lambda e: self.face_picked())
        self.face_info = ttk.Label(f, wraplength=350, justify="left")
        self.face_info.pack(fill="x", pady=(0, 6))
        self.face_vars = []
        for label in ('Rotation X', 'Rotation Y', 'Rotation Z', 'Translation X', 'Translation Y', 'Translation Z'):
            row = ttk.Frame(f)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, width=13).pack(side="left")
            var = tk.DoubleVar(value=0.0)
            self.face_vars.append(var)
            limit = 60 if len(self.face_vars) <= 3 else 0.2
            ttk.Scale(row, from_=-limit, to=limit, variable=var,
                      command=lambda value: self.face_changed()).pack(side="left", fill="x", expand=True)
            spin = ttk.Spinbox(row, from_=-limit * 10, to=limit * 10, increment=1 if limit == 60 else .005,
                              width=7, textvariable=var, command=self.face_changed)
            spin.pack(side="left", padx=(4, 0))
            spin.bind("<Return>", lambda e: self.face_changed())
            spin.bind("<FocusOut>", lambda e: self.face_changed())
        row = ttk.Frame(f)
        row.pack(fill="x", pady=8)
        ttk.Button(row, text='Reset bone', command=self.zero_face_bone).pack(side="left")
        ttk.Button(row, text='Reset all', command=self.zero_face).pack(side="left", padx=4)

    def fill_face(self) -> None:
        bones = self.model.bones if self.model else []
        self.face_indices = igb_model.facial_bones(bones) if self.face_only.get() else list(range(len(bones)))
        self.face_choice.configure(values=[f"{i}: {bones[i].name}" for i in self.face_indices])
        if self.face_indices:
            self.face_choice.current(0)
            self.face_picked(highlight=False)
        else:
            self.face_choice.set("")
            self.face_info.configure(text='No facial bones. Disable the filter to see all bones.'
                                     if bones else 'This file contains no skeleton.')

    def face_picked(self, highlight=True) -> None:
        choice = self.face_choice.current()
        if not 0 <= choice < len(self.face_indices):
            return
        index = self.face_indices[choice]
        self.updating_face = True
        try:
            for var, value in zip(self.face_vars, self.bone_offsets.get(index, (0,) * 6)):
                var.set(value)
        finally:
            self.updating_face = False
        self.face_info.configure(text=tr('{0}\nRotations in degrees; translations in model units. Children follow the bone.', self.model.bones[index].name))
        if highlight:
            self.heat_var.set(False)
            self.select_bone(index)

    def face_changed(self) -> None:
        choice = self.face_choice.current()
        if self.updating_face or not 0 <= choice < len(self.face_indices):
            return
        try:
            values = tuple(v.get() for v in self.face_vars)
        except (tk.TclError, ValueError):
            return  # a spinbox can be temporarily empty while typing
        if not all(math.isfinite(v) for v in values):
            return
        index = self.face_indices[choice]
        if any(values):
            self.bone_offsets[index] = values
        else:
            self.bone_offsets.pop(index, None)
        self.redraw()

    def zero_face_bone(self) -> None:
        choice = self.face_choice.current()
        if 0 <= choice < len(self.face_indices):
            self.bone_offsets.pop(self.face_indices[choice], None)
            self.face_picked()
            self.redraw()

    def zero_face(self) -> None:
        self.bone_offsets.clear()
        self.face_picked()
        self.redraw()

    def build_textures(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb, padding=6)
        self.texture_tab = f
        nb.add(f, text="Texture")
        self.tex_list = tk.Listbox(f, height=6, exportselection=False)
        self.tex_list.pack(fill="x")
        self.tex_list.bind("<<ListboxSelect>>", lambda e: self.tex_picked())
        self.tex_preview = tk.Label(f, anchor="center", background="#272933", borderwidth=0)
        self.tex_preview.pack(fill="both", expand=True, pady=6)
        self.tex_info = ttk.Label(f, text="", wraplength=330, justify="left", foreground="#444")
        self.tex_info.pack(fill="x")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text='3D filtering:').pack(side="left")
        self.filter_var_3d = tk.StringVar(value=tr('Linear'))
        sampling = ttk.Combobox(row, textvariable=self.filter_var_3d, values=(tr('Linear'), "Pixel"),
                                state="readonly", width=9)
        sampling.pack(side="left", padx=4)
        sampling.bind("<<ComboboxSelected>>", lambda e: self.view.filter_changed())
        ttk.Button(row, text='Enlarge…', command=self.preview_texture).pack(side="left")
        self.tex_preview.bind("<Double-Button-1>", lambda e: self.preview_texture())
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text='Export PNG…', command=self.export_png).pack(side="left")
        ttk.Button(row, text='Replace with PNG…', command=self.replace_png).pack(side="left", padx=(4, 0))

    def status(self, text: str) -> None:
        self.status_label.configure(text=i18n.tr(text))

    def redraw(self) -> None:
        if hasattr(self, "view"):
            self.view.draw()

    # Opening.

    def ask_open(self) -> None:
        path = filedialog.askopenfilename(parent=self, title='Open a model',
                                          initialdir=self.settings.get("igb_dir") or None,
                                          filetypes=[('Models, archives and images', "*.igb *.gfc *.gob *.glb *.gltf *.png *.tga"), ("Alchemy .igb", "*.igb"),
                                                     ('CNK archives', "*.gfc *.gob"),
                                                     ("glTF", "*.glb *.gltf"), ('All files', "*.*")])
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path) -> None:
        self.settings["igb_dir"] = str(path.parent)
        save_settings(self.settings)
        if path.suffix.lower() in (".gfc", ".gob"):
            self.open_gob(path)
            return
        if path.suffix.lower() in (".glb", ".gltf"):
            self.open_gltf_alone(path)
            return
        try:
            data = path.read_bytes()
        except OSError as e:
            messagebox.showerror('Open', tr('Could not read the file: {0}', e), parent=self)
            return
        if path.suffix.lower() == ".igb":
            self.load_igb(data, path.name, str(path))
        else:
            self.preview_asset(data, path.name)

    def load_igb(self, data: bytes, name: str, where: str, platform: str | None = None) -> bool:
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            model = igb_model.from_igb(data, name, platform)
        except (igb_file.IgbError, igb_model.ModelError, ValueError, KeyError, IndexError) as e:
            messagebox.showerror('Open', tr('{0}: could not read this IGB ({1}).', name, e), parent=self)
            return False
        finally:
            self.config(cursor="")
        self.base, self.base_data, self.base_name, self.imported = model, data, name, None
        self.tex_changes = {}
        self.show(model)
        self.title_label.configure(text=f"{name}  —  {where}")
        self.status(tr('Opened {0}: {1} bones, {2} meshes, {3} textures, {4} animations.', name, len(model.bones), len(model.meshes), len(model.textures), len(model.clips)))
        return True

    def open_gltf_alone(self, path: Path) -> None:
        try:
            model = igb_gltf.load(path)
        except (igb_gltf.GltfError, ValueError, KeyError, IndexError, OSError) as e:
            messagebox.showerror('Open', tr('{0}: could not read glTF ({1}).', path.name, e), parent=self)
            return
        self.base = self.base_data = None
        self.imported = None
        self.base_name = ""
        self.tex_changes = {}
        self.show(model)
        self.title_label.configure(text=f"{path.name}  —  {path}")
        self.status(tr('Opened {0}: {1} bones, {2} meshes, {3} blendshapes, {4} animations. To write to an IGB, open that IGB first, then choose “Import glTF…”.', path.name, len(model.bones), len(model.meshes), sum((len(m.shapes) for m in model.meshes)), len(model.clips)))

    def show(self, model: igb_model.Model) -> None:
        """A model in the view and the tabs."""
        self.model = model
        self.bone_offsets.clear()
        self.view.reset()
        self.bone = None
        self.clip = None
        self.time = 0.0
        self.playing = False
        self.play_button.configure(text="▶ Play")
        best = min((m.lod for m in model.meshes), default=0)
        self.hidden = {k for k, m in enumerate(model.meshes) if m.lod != best}
        if len(model.meshes) > 12:  # a track: its sky dome hides it from outside
            self.hidden |= {k for k, m in enumerate(model.meshes) if "sky" in m.name.lower()}
        self.fill_meshes()
        self.fill_bones()
        self.fill_anims()
        self.fill_face()
        self.fill_shapes()
        self.fill_textures()
        self.inspector.set_file(model.igb or (self.base.igb if self.base else None), model.notes)
        if not model.meshes and model.textures:
            self.tabs.select(self.texture_tab)
        self.undo_button.configure(state="normal" if self.imported is not None else "disabled")
        self.save_button.configure(state="normal" if self.base is not None and
                                   (self.imported is not None or self.tex_changes) else "disabled")
        self.view.frame_model()

    # The archive.

    def ask_gob(self) -> None:
        path = filedialog.askopenfilename(parent=self, title='Open a Crash Nitro Kart archive',
                                          initialdir=str(Path(self.settings["igb_gfc"]).parent)
                                          if self.settings.get("igb_gfc") else None,
                                          filetypes=[("ASSETS.GFC / GOB", "*.gfc *.GFC *.gob *.GOB"), ('All files', "*.*")])
        if path:
            self.open_gob(Path(path))

    def open_gob(self, path: Path, quiet: bool = False) -> None:
        if path.suffix.lower() == ".gob":
            path = path.with_suffix(".gfc")
        try:
            self.gob = cnk_gob.Gob(path)
        except (OSError, cnk_gob.GobError, ValueError) as e:
            if not quiet:
                messagebox.showerror('Archive', tr('Could not open {0}: {1}', path.name, e), parent=self)
            return
        self.settings["igb_gfc"] = str(path)
        save_settings(self.settings)
        n = len(self.gob.igbs())
        self.gob_label.configure(text=tr('{0}\n{1} files, {2} IGBs. Double-click to open.', path, len(self.gob.files), n))
        self.fill_gob()
        if not quiet:
            self.tabs.select(0)

    def fill_gob(self) -> None:
        if self.gob is None:
            return
        flt = self.filter_var.get().strip().lower()
        self.settings["igb_filter"] = self.filter_var.get()
        self.settings["archive_browser_v2"] = True
        category = cnk_assets.CATEGORIES[max(0, self.category_choice.current())]
        self.gob_entries = sorted((e for e in self.gob.files if cnk_assets.category_match(e.path, category)
                                   and all(w in e.path.lower() for w in flt.split())), key=lambda e: e.path.lower())
        self.archive_count.configure(text=tr('{0} / {1} files visible', len(self.gob_entries), len(self.gob.files)))
        self.gob_list.delete(0, "end")
        for e in self.gob_entries:
            raw_insert(self.gob_list, f"{e.path}   ({e.size // 1024} KB)")

    def gob_selected(self) -> cnk_gob.Entry | None:
        sel = self.gob_list.curselection()
        return self.gob_entries[sel[0]] if sel and sel[0] < len(self.gob_entries) else None

    def open_from_gob(self) -> None:
        e = self.gob_selected()
        if e is None:
            return
        try:
            data = self.gob.read(e)
        except (OSError, cnk_gob.GobError, ValueError) as ex:
            messagebox.showerror('Archive', f"{e.path}: {ex}", parent=self)
            return
        if Path(e.path).suffix.lower() != ".igb":
            self.preview_asset(data, e.path)
        elif self.load_igb(data, Path(e.path).name, f"ASSETS.GOB: {e.path}", self.gob.platform):
            self.base_where = e.path

    def preview_asset(self, data: bytes, name: str) -> None:
        window = tk.Toplevel(self)
        window.title(name)
        if Path(name).suffix.lower() in (".png", ".tga"):
            try:
                w, h, rgba = cnk_assets.image(data, name)
            except (ValueError, IndexError, KeyError) as ex:
                window.destroy()
                messagebox.showerror('Image', str(ex), parent=self)
                return
            window.preview = photo(w, h, rgba, min(800, self.winfo_screenheight() - 180))
            tk.Label(window, image=window.preview, background="#272933").pack(fill="both", expand=True,
                                                                                 padx=8, pady=8)
            ttk.Label(window, text=f"{w} × {h} · {len(data)} byte").pack(padx=8, pady=(0, 8))
        else:
            ttk.Label(window, text=tr('{0}\n{1} bytes · data preview; full extraction available', name, len(data)),
                      wraplength=740).pack(fill="x", padx=8, pady=8)
            box = tk.Text(window, width=95, height=25, wrap="none", font=("Consolas", 10))
            box.pack(fill="both", expand=True, padx=8)
            if Path(name).suffix.lower() in (".csv", ".ini", ".cfg", ".opt", ".sln", ".int", ".rcf"):
                try:
                    text = data[:65536].decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = data[:65536].decode("latin-1")
            else:
                text = "\n".join(f"{i:08X}  {data[i:i + 16].hex(' ')}" for i in range(0, min(len(data), 4096), 16))
            box.insert("1.0", text)
            box.configure(state="disabled")
        def extract():
            path = filedialog.asksaveasfilename(parent=window, initialfile=Path(name).name,
                                               defaultextension=Path(name).suffix)
            if path:
                try:
                    Path(path).write_bytes(data)
                except OSError as ex:
                    messagebox.showerror('Extract', str(ex), parent=window)
        ttk.Button(window, text='Extract file…', command=extract).pack(padx=8, pady=8)

    def extract_from_gob(self) -> None:
        e = self.gob_selected()
        if e is None:
            return
        path = filedialog.asksaveasfilename(parent=self, title='Extract the file', initialfile=Path(e.path).name,
                                            initialdir=self.settings.get("igb_dir") or None,
                                            defaultextension=Path(e.path).suffix, filetypes=[('All files', "*.*")])
        if path:
            try:
                Path(path).write_bytes(self.gob.read(e))
            except (OSError, cnk_gob.GobError) as ex:
                messagebox.showerror('Extract', str(ex), parent=self)
                return
            self.status(tr('Extracted {0}.', Path(path).name))

    # Meshes.

    def visible(self, m: igb_model.Mesh) -> bool:
        model = self.model
        if model is None:
            return False
        try:
            k = model.meshes.index(m)
        except ValueError:
            return False
        return k not in self.hidden

    def fill_meshes(self) -> None:
        t = self.mesh_tree
        t.delete(*t.get_children())
        model = self.model
        for k, m in enumerate(model.meshes if model else []):
            if m.skinned:
                kind = i18n.tr("skin")
            elif m.bone >= 0:
                kind = i18n.tr('on a bone')
            else:
                kind = i18n.tr('rigid')
            if m.lod == igb_model.SHADOW:
                kind = i18n.tr('shadow')
            elif m.lod:
                kind += f", LOD {m.lod}"
            t.insert("", "end", iid=str(k), values=("☐" if k in self.hidden else "☑", len(m.pos), len(m.tris), kind))
            raw_text(t, str(k), m.name)

    def mesh_click(self, e) -> None:
        row = self.mesh_tree.identify_row(e.y)
        if not row:
            return
        k = int(row)
        m = self.model.meshes[k]
        if self.mesh_tree.identify_column(e.x) == "#1":
            self.hidden ^= {k}
            self.mesh_tree.set(row, "vis", "☐" if k in self.hidden else "☑")
            self.redraw()
        tex = self.model.textures[m.texture].name if 0 <= m.texture < len(self.model.textures) else \
            i18n.tr('none')
        lines = [m.name, tr('Texture: {0}', tex)]
        if m.skinned:
            lines.append(tr('Skinned to {0} bones', len({b for vw in m.weights for b, w in vw if w > 0})))
        elif 0 <= m.bone < len(self.model.bones):
            lines.append(tr('Follows bone {0}', self.model.bones[m.bone].name))
        else:
            lines.append(i18n.tr('Fixed (does not follow bones)'))
        if m.shapes:
            lines.append(tr('Blendshapes: {0}', len(m.shapes)))
        raw_label(self.mesh_info, "\n".join(lines))

    def show_meshes(self, everything: bool) -> None:
        if self.model is None:
            return
        best = min((m.lod for m in self.model.meshes), default=0)
        self.hidden = set() if everything else {k for k, m in enumerate(self.model.meshes) if m.lod != best}
        self.fill_meshes()
        self.redraw()

    # Bones.

    def fill_bones(self) -> None:
        t = self.bone_tree
        t.delete(*t.get_children())
        model = self.model
        if model is None:
            return
        for i, b in enumerate(model.bones):
            parent = str(b.parent) if 0 <= b.parent < i else ""
            t.insert(parent, "end", iid=str(i), values=(b.bm if b.bm >= 0 else "",), open=True)
            raw_text(t, str(i), b.name)
        self.bone_info.configure(text=tr('{0} bones. Select one here or click its joint in the view.', len(model.bones)) if model.bones else 'This model has no bones.')

    def bone_picked(self, _e=None) -> None:
        sel = self.bone_tree.selection()
        if not sel:
            return
        i = int(sel[0])
        if i == self.bone:
            return
        self.bone = i
        self.describe_bone()
        self.redraw()

    def select_bone(self, i: int) -> None:
        self.bone = i
        self.bone_tree.selection_set(str(i))
        self.bone_tree.see(str(i))
        self.describe_bone()
        self.redraw()

    def describe_bone(self) -> None:
        model, i = self.model, self.bone
        if model is None or i is None or i >= len(model.bones):
            return
        b = model.bones[i]
        verts = 0
        for m in model.meshes:
            if m.skinned:
                verts += sum(1 for vw in m.weights if any(k == i and w > 0 for k, w in vw))
        parent = model.bones[b.parent].name if 0 <= b.parent < len(model.bones) else i18n.tr('None')
        kids = [c.name for c in model.bones if c.parent == i]
        p = b.bind[12:15]
        lines = [b.name, tr('Parent: {0}', parent), tr('Children: {0}', len(kids)),
                 tr('Moves {0} vertices', verts), tr('Rest position: {0:.3f}, {1:.3f}, {2:.3f}', p[0], p[1], p[2])]
        if b.bm < 0 and model.igb is not None:
            lines.append(i18n.tr('No skinning matrix: this is a helper (IK or attachment) and does not deform the skin.'))
        raw_label(self.bone_info, "\n".join(lines))
        self.status(tr('Selected bone: {0}', b.name))

    # Animations.

    def fill_anims(self) -> None:
        self.anim_list.delete(0, "end")
        for c in self.model.clips if self.model else []:
            raw_insert(self.anim_list, f"{c.name}   ({c.length:.2f} s)")
        self.time_label.configure(text='Animated node path and axes; select a clip.'
                                  if self.model and self.model.scene_motion and not self.model.meshes
                                  and not self.model.bones else 'rest pose')

    def anim_picked(self) -> None:
        sel = self.anim_list.curselection()
        if not sel:
            return
        self.clip = sel[0]
        self.time = 0.0
        c = self.model.clips[self.clip]
        self.time_scale.configure(to=max(0.01, c.length))
        self.time_var.set(0.0)
        if not self.playing:
            self.toggle_play()
        self.view.follow()

    def toggle_play(self) -> None:
        if self.clip is None and self.model and self.model.clips:
            self.anim_list.selection_set(0)
            self.anim_picked()
            return
        self.playing = not self.playing and self.clip is not None
        self.play_button.configure(text='⏸ Pause' if self.playing else "▶ Play")
        self.last_tick = time.perf_counter()

    def rest_pose(self) -> None:
        self.clip = None
        self.playing = False
        self.play_button.configure(text="▶ Play")
        self.anim_list.selection_clear(0, "end")
        self.time_label.configure(text='rest pose')
        self.view.follow()

    def scrub(self, value) -> None:
        if self.clip is None:
            return
        self.time = float(value)
        self.update_time_label()
        if not self.playing:
            self.redraw()

    def update_time_label(self) -> None:
        if self.clip is not None and self.model:
            c = self.model.clips[self.clip]
            self.time_label.configure(text=f"{c.name}: {self.time:.2f} / {c.length:.2f} s")

    def pose(self) -> list | None:
        model = self.model
        if model is None or not model.bones:
            return []
        if self.clip is None or self.clip >= len(model.clips):
            return igb_model.offset_pose(model.bones, [b.bind for b in model.bones], self.bone_offsets)
        try:
            return igb_model.offset_pose(model.bones, model.clips[self.clip].pose(self.time), self.bone_offsets)
        except (IndexError, ValueError, ZeroDivisionError):
            return [b.bind for b in model.bones]

    def tick(self) -> None:
        now = time.perf_counter()
        dt, self.last_tick = min(0.1, now - self.last_tick), now
        if self.playing and self.clip is not None and self.model:
            c = self.model.clips[self.clip]
            speed = float(self.speed_var.get().rstrip("×") or 1)
            self.time += dt * speed
            if self.time > c.length:
                self.time = 0.0 if c.length <= 0 else self.time % c.length
            self.time_var.set(self.time)
            self.update_time_label()
            self.redraw()
        self.tick_id = self.after(33, self.tick)

    # Blend shapes.

    def fill_shapes(self) -> None:
        for w in self.shape_frame.winfo_children():
            w.destroy()
        self.shape_vars = {}
        model = self.model
        with_shapes = [(k, m) for k, m in enumerate(model.meshes if model else []) if m.shapes]
        if not with_shapes:
            ttk.Label(self.shape_frame, text=NO_SHAPES, wraplength=320, justify="left").pack(fill="x", padx=4,
                                                                                            pady=4)
            return
        for k, m in with_shapes:
            ttk.Label(self.shape_frame, text=f"{m.name} ({len(m.shapes)})", style="Head.TLabel").pack(
                fill="x", padx=4, pady=(6, 2))
            start = list(getattr(m, "default_weights", []) or [])
            vars_ = []
            for s, (name, _) in enumerate(m.shapes):
                row = ttk.Frame(self.shape_frame)
                row.pack(fill="x", padx=4)
                ttk.Label(row, text=name, width=18).pack(side="left")
                v = tk.DoubleVar(value=start[s] if s < len(start) else 0.0)
                val = ttk.Label(row, text=f"{v.get():.2f}", width=5)
                ttk.Scale(row, from_=0.0, to=1.0, variable=v, length=180,
                          command=lambda x, lab=val: (lab.configure(text=f"{float(x):.2f}"), self.redraw())
                          ).pack(side="left", padx=4)
                val.pack(side="left")
                vars_.append(v)
            self.shape_vars[k] = vars_

    def shape_weights(self, k: int) -> list[float]:
        return [v.get() for v in self.shape_vars.get(k, [])]

    def zero_shapes(self) -> None:
        for vs in self.shape_vars.values():
            for v in vs:
                v.set(0.0)
        self.fill_shapes_labels()
        self.redraw()

    def fill_shapes_labels(self) -> None:
        for row in self.shape_frame.winfo_children():
            kids = row.winfo_children() if isinstance(row, ttk.Frame) else []
            if len(kids) == 3:
                kids[2].configure(text="0.00")

    # Textures.

    def fill_textures(self) -> None:
        self.tex_list.delete(0, "end")
        for k, t in enumerate(self.model.textures if self.model else []):
            mark = "  *" if k in self.tex_changes else ""
            raw_insert(self.tex_list, f"{t.name}   {t.w}×{t.h}{mark}")
        self.tex_preview.configure(image="")
        self.tex_info.configure(text="")
        if self.model and self.model.textures:
            self.tex_list.selection_set(0)
            self.tex_picked()

    def tex_picked(self) -> None:
        sel = self.tex_list.curselection()
        if not sel or self.model is None:
            return
        t = self.model.textures[sel[0]]
        if not self.model.meshes:
            self.view.show_image(t)
        try:
            self.images["tex"] = photo(t.w, t.h, t.rgba, 300)
            self.tex_preview.configure(image=self.images["tex"])
        except tk.TclError:
            self.tex_preview.configure(image="")
        users = [m.name for m in self.model.meshes if m.texture == sel[0]]
        lines = [f"{t.name}: {t.w}×{t.h}", tr('Used by: {0}', ', '.join(users)) if users else
                 i18n.tr('Not used by any mesh')]
        if self.model is self.base and t.image >= 0:
            img = self.base.igb.objects[t.image]
            bits = self.base.igb.get(img, igb_model.IM_BITS, 0)
            fmt = self.base.igb.get(img, igb_model.IM_FMT, 0)
            encoding = {0: "L8", 8: "RGBA5551", 14: "DXT1", 15: "DXT3", 16: "DXT5"}.get(fmt)
            lines.append(tr('Stored as: {0}', encoding) if encoding else
                         i18n.tr('Stored as: 256 colors' if bits == 8 else 'Stored as: 16 colors' if bits == 4 else
                                 'Stored as: RGB(A)'))
        raw_label(self.tex_info, "\n".join(lines))

    def preview_texture(self):
        sel = self.tex_list.curselection()
        if not sel or self.model is None:
            return None
        t = self.model.textures[sel[0]]
        window = tk.Toplevel(self)
        window.title(tr('{0} — {1}×{2} original pixels', t.name, t.w, t.h))
        window.geometry("760x650")
        toolbar = ttk.Frame(window, padding=6)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text=tr('Original: {0}×{1} · Zoom:', t.w, t.h)).pack(side="left")
        zoom = tk.StringVar(value="400%" if max(t.w, t.h) <= 256 else "100%")
        choices = ttk.Combobox(toolbar, textvariable=zoom, values=("100%", "200%", "400%", "800%"),
                               state="readonly", width=7)
        choices.pack(side="left", padx=4)
        ttk.Label(toolbar, text='Original pixels enlarged').pack(side="left", padx=6)
        body = ttk.Frame(window)
        body.pack(fill="both", expand=True)
        canvas = tk.Canvas(body, background="#272933", highlightthickness=0)
        vertical = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        horizontal = ttk.Scrollbar(body, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        item = canvas.create_image(0, 0, anchor="nw")
        def refresh(event=None):
            factor = int(zoom.get().rstrip("%")) // 100
            window.preview = photo(t.w, t.h, t.rgba, max(t.w, t.h) * factor)
            canvas.itemconfigure(item, image=window.preview)
            canvas.configure(scrollregion=(0, 0, window.preview.width(), window.preview.height()))
        choices.bind("<<ComboboxSelected>>", refresh)
        refresh()
        return window

    def export_png(self) -> None:
        sel = self.tex_list.curselection()
        if not sel or self.model is None:
            return
        t = self.model.textures[sel[0]]
        path = filedialog.asksaveasfilename(parent=self, title='Export the texture', defaultextension=".png",
                                            initialfile=Path(t.name).stem + ".png",
                                            initialdir=self.settings.get("igb_out") or None,
                                            filetypes=[("PNG", "*.png")])
        if path:
            Path(path).write_bytes(png_util.png_bytes(t.w, t.h, t.rgba))
            self.settings["igb_out"] = str(Path(path).parent)
            save_settings(self.settings)
            self.status(tr('Exported {0}.', Path(path).name))

    def replace_png(self) -> None:
        sel = self.tex_list.curselection()
        if not sel or self.model is None:
            return
        if self.base is None:
            messagebox.showinfo("Texture", 'Open an IGB to replace its textures.', parent=self)
            return
        k = sel[0]
        t = self.model.textures[k]
        path = filedialog.askopenfilename(parent=self, title='Replace the texture with a PNG',
                                          initialdir=self.settings.get("igb_out") or None,
                                          filetypes=[("PNG", "*.png")])
        if not path:
            return
        try:
            w, h, rgba = igb_gltf.rgba_of(Path(path).read_bytes(), Path(path).name)
        except (OSError, ValueError, TypeError) as e:
            messagebox.showerror("Texture", tr('Could not read PNG: {0}', e), parent=self)
            return
        target = k
        if self.model is not self.base:  # an imported model: the base's texture with that name
            target = next((j for j, b in enumerate(self.base.textures) if igb_model.same_texture(b.name, t.name)),
                          -1)
            if target < 0:
                messagebox.showerror("Texture", tr('No texture named “{0}” in the IGB.', t.name), parent=self)
                return
        bt = self.base.textures[target]
        if (w, h) != (bt.w, bt.h):
            if not messagebox.askokcancel("Texture", tr('The PNG is {0}×{1} and the game texture is {2}×{3}. Resize it to match?', w, h, bt.w, bt.h), parent=self):
                return
        self.tex_changes[target] = (w, h, rgba)
        # What it will look like: through the file's own palette and size.
        try:
            data, _ = igb_model.with_textures(self.base, {target: (w, h, rgba)})
            look = igb_model.from_igb(data, self.base_name, self.base.platform).textures[target]
            t.w, t.h, t.rgba = look.w, look.h, look.rgba
        except (igb_model.ModelError, igb_file.IgbError, ValueError) as e:
            del self.tex_changes[target]
            messagebox.showerror("Texture", tr('Could not write the texture: {0}', e), parent=self)
            return
        self.view.drop_texture(k)
        self.fill_textures()
        self.tex_list.selection_clear(0, "end")
        self.tex_list.selection_set(k)
        self.tex_picked()
        self.save_button.configure(state="normal")
        self.redraw()
        self.status(tr("Texture {0} replaced (converted to the game's colors). Choose “Save .igb…” to write it.", t.name))

    # Out.

    def export_gltf(self) -> None:
        if self.model is None:
            return
        stem = Path(self.base_name or self.model.source or 'model').stem
        path = filedialog.asksaveasfilename(parent=self, title='Export to glTF', defaultextension=".glb",
                                            initialfile=stem + ".glb",
                                            initialdir=self.settings.get("igb_out") or None,
                                            filetypes=[('Binary glTF', "*.glb"), ('glTF (text)', "*.gltf")])
        if not path:
            return
        meshes = [m for k, m in enumerate(self.model.meshes) if k not in self.hidden]
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            igb_gltf.save(self.model, Path(path), meshes)
        except (OSError, ValueError) as e:
            messagebox.showerror('Export', str(e), parent=self)
            return
        finally:
            self.config(cursor="")
        self.settings["igb_out"] = str(Path(path).parent)
        save_settings(self.settings)
        self.status(tr('Exported {0}: {1} meshes, {2} bones, {3} animations. In Blender: File > Import > glTF 2.0.', Path(path).name, len(meshes), len(self.model.bones), len(self.model.clips)))

    def export_obj(self) -> None:
        if self.model is None:
            return
        stem = Path(self.base_name or self.model.source or 'model').stem
        path = filedialog.asksaveasfilename(parent=self, title='Export to OBJ', defaultextension=".obj",
                                            initialfile=stem + ".obj",
                                            initialdir=self.settings.get("igb_out") or None,
                                            filetypes=[("Wavefront OBJ", "*.obj")])
        if not path:
            return
        shown = igb_model.Model()
        shown.__dict__.update(self.model.__dict__)
        shown.meshes = [m for k, m in enumerate(self.model.meshes) if k not in self.hidden]
        igb_model.write_obj(shown, Path(path))
        self.status(tr('Exported {0} (geometry and texture coordinates only).', Path(path).name))

    # In.

    def import_gltf(self) -> None:
        if self.base is None:
            if messagebox.askyesno('Import glTF', 'No IGB is open. Open this file for viewing? To write to an IGB, open that IGB first.', parent=self):
                path = filedialog.askopenfilename(parent=self, title='Open a glTF',
                                                  initialdir=self.settings.get("igb_out") or None,
                                                  filetypes=[("glTF", "*.glb *.gltf")])
                if path:
                    self.open_gltf_alone(Path(path))
            return
        path = filedialog.askopenfilename(parent=self, title='Import a glTF into the model',
                                          initialdir=self.settings.get("igb_out") or None,
                                          filetypes=[("glTF", "*.glb *.gltf")])
        if not path:
            return
        try:
            other = igb_gltf.load(Path(path))
        except (igb_gltf.GltfError, ValueError, KeyError, IndexError, OSError) as e:
            messagebox.showerror('Import', tr('Could not read glTF: {0}', e), parent=self)
            return
        pairs = igb_model.match_meshes(self.base, other)
        if not pairs:
            messagebox.showerror('Import', tr('No glTF mesh matches a mesh name in the IGB ({0}): use the names from the export.',
                                            ", ".join(m.name for m in self.base.meshes[:6])), parent=self)
            return
        # The bones the file does not have, and what the save will do.
        names = {n.lower() for n in igb_model.unique_names([b.name for b in self.base.bones])}
        extra = [b.name for b in other.bones if b.name.split("|")[-1].lower() not in names]
        self.imported = other
        other.source = Path(path).name
        self.show(other)
        msg = [tr('Imported {0} onto {1}: {2} meshes to replace ({3}).', Path(path).name, self.base_name, len(pairs), ', '.join((a.name for a, _ in pairs)))]
        if extra:
            msg.append(tr('Bones missing from the IGB (their weights go to the parent): {0}.', ', '.join(extra[:6])))
        if any(m.shapes for m in other.meshes):
            msg.append(i18n.tr('Blendshapes can be previewed but cannot be stored in the IGB.'))
        msg.append(i18n.tr('Choose “Save .igb…” to write it.'))
        raw_label(self.status_label, " ".join(msg))

    def drop_import(self) -> None:
        if self.base is None:
            return
        self.imported = None
        self.show(self.base)
        self.status(tr('Restored the original IGB {0}.', self.base_name))

    def save_igb(self) -> None:
        if self.base is None:
            return
        notes: list[str] = []
        try:
            data = self.base_data
            if self.imported is not None:
                data, notes = igb_model.apply(self.base, self.imported)
            if self.tex_changes:
                again = igb_model.from_igb(data, self.base_name, self.base.platform)
                data, more = igb_model.with_textures(again, self.tex_changes)
                notes += more
            check = igb_model.from_igb(data, self.base_name, self.base.platform)  # read back: it must open
        except (igb_model.ModelError, igb_file.IgbError, ValueError, KeyError, IndexError) as e:
            messagebox.showerror('Save', tr('Could not write the IGB: {0}', e), parent=self)
            return
        path = filedialog.asksaveasfilename(parent=self, title='Save the IGB', defaultextension=".igb",
                                            initialfile=self.base_name,
                                            initialdir=self.settings.get("igb_out") or None,
                                            filetypes=[("Alchemy .igb", "*.igb")])
        if not path:
            return
        try:
            Path(path).write_bytes(data)
        except OSError as e:
            messagebox.showerror('Save', str(e), parent=self)
            return
        self.settings["igb_out"] = str(Path(path).parent)
        save_settings(self.settings)
        size = f"{len(self.base_data) // 1024} → {len(data) // 1024} KB"
        messagebox.showinfo('Save', tr('Wrote {0} ({1}, {2} meshes).\n\n', Path(path).name, size, len(check.meshes))
                            + "\n".join(notes[:14]), parent=self)
        self.status(tr('Wrote {0} ({1}).', Path(path).name, size))

    # The rest.

    def help(self) -> None:
        messagebox.showinfo('IGB Viewer', HELP, parent=self)

    def reopen(self) -> bool:
        self.destroy()
        i18n.use()
        Viewer(None).mainloop()
        return True

    def quit_viewer(self) -> None:
        save_settings(self.settings)
        self.destroy()

    def destroy(self) -> None:
        tick_id = getattr(self, "tick_id", None)
        if tick_id is not None:
            self.after_cancel(tick_id)
            self.tick_id = None
        super().destroy()


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    i18n.install()
    Viewer(path).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
