"""OpenGL 1.1 and WGL through ctypes (opengl32 on Windows): the entry points and constants the 3D view uses, and the
pixel format a Tk frame's window gets. (Copied from the Crash WoC editor's gl_view.py, so that this tool stands on
its own.)"""
import ctypes

GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT = 0x4000, 0x100
GL_PROJECTION, GL_MODELVIEW = 0x1701, 0x1700
GL_DEPTH_TEST, GL_BLEND, GL_TEXTURE_2D, GL_ALPHA_TEST, GL_CULL_FACE = 0x0B71, 0x0BE2, 0x0DE1, 0x0BC0, 0x0B44
GL_POLYGON_OFFSET_LINE = 0x2A02
GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO = 0x0302, 0x0303, 1, 0
GL_ONE_MINUS_DST_COLOR = 0x0307
GL_GREATER, GL_LEQUAL, GL_ALWAYS = 0x0204, 0x0203, 0x0207
GL_TRIANGLES, GL_LINES, GL_LINE_LOOP, GL_QUADS = 4, 1, 2, 7
GL_FLOAT, GL_UNSIGNED_BYTE, GL_UNSIGNED_INT = 0x1406, 0x1401, 0x1405
GL_VERTEX_ARRAY, GL_COLOR_ARRAY, GL_TEXTURE_COORD_ARRAY = 0x8074, 0x8076, 0x8078
GL_RGBA, GL_RGB = 0x1908, 0x1907
GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR, GL_NEAREST = 0x2801, 0x2800, 0x2601, 0x2600
GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_REPEAT = 0x2802, 0x2803, 0x2901
GL_FRONT_AND_BACK, GL_LINE, GL_FILL = 0x0408, 0x1B01, 0x1B02
GL_UNPACK_ALIGNMENT, GL_PACK_ALIGNMENT = 0x0CF5, 0x0D05
GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_TEXTURE_ENV_COLOR, GL_MODULATE = 0x2300, 0x2200, 0x2201, 0x2100
GL_COMBINE, GL_COMBINE_RGB, GL_COMBINE_ALPHA, GL_REPLACE = 0x8570, 0x8571, 0x8572, 0x1E01
GL_SOURCE0_RGB, GL_SOURCE0_ALPHA, GL_SOURCE1_ALPHA = 0x8580, 0x8588, 0x8589
GL_CONSTANT, GL_PRIMARY_COLOR, GL_TEXTURE = 0x8576, 0x8577, 0x1702
GL_SCISSOR_TEST, GL_CLAMP_TO_EDGE = 0x0C11, 0x812F


class GL:
    """The few OpenGL 1.1 and WGL entry points the view uses."""
    _sigs = {
        "glViewport": [ctypes.c_int] * 4, "glClearColor": [ctypes.c_float] * 4, "glClear": [ctypes.c_uint],
        "glMatrixMode": [ctypes.c_uint], "glLoadIdentity": [], "glLoadMatrixf": [ctypes.c_void_p],
        "glMultMatrixf": [ctypes.c_void_p], "glPushMatrix": [], "glPopMatrix": [],
        "glFrustum": [ctypes.c_double] * 6, "glEnable": [ctypes.c_uint], "glDisable": [ctypes.c_uint],
        "glBlendFunc": [ctypes.c_uint] * 2, "glAlphaFunc": [ctypes.c_uint, ctypes.c_float],
        "glDepthMask": [ctypes.c_ubyte], "glDepthFunc": [ctypes.c_uint],
        "glGenTextures": [ctypes.c_int, ctypes.c_void_p], "glBindTexture": [ctypes.c_uint] * 2,
        "glTexImage2D": [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                         ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p],
        "glTexParameteri": [ctypes.c_uint, ctypes.c_uint, ctypes.c_int],
        "glDeleteTextures": [ctypes.c_int, ctypes.c_void_p],
        "glEnableClientState": [ctypes.c_uint], "glDisableClientState": [ctypes.c_uint],
        "glVertexPointer": [ctypes.c_int, ctypes.c_uint, ctypes.c_int, ctypes.c_void_p],
        "glTexCoordPointer": [ctypes.c_int, ctypes.c_uint, ctypes.c_int, ctypes.c_void_p],
        "glColorPointer": [ctypes.c_int, ctypes.c_uint, ctypes.c_int, ctypes.c_void_p],
        "glDrawElements": [ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p],
        "glColor4f": [ctypes.c_float] * 4, "glColor3ub": [ctypes.c_ubyte] * 3, "glBegin": [ctypes.c_uint],
        "glEnd": [], "glVertex3f": [ctypes.c_float] * 3, "glLineWidth": [ctypes.c_float],
        "glPolygonMode": [ctypes.c_uint] * 2, "glPolygonOffset": [ctypes.c_float] * 2,
        "glPixelStorei": [ctypes.c_uint, ctypes.c_int], "glTexEnvi": [ctypes.c_uint, ctypes.c_uint, ctypes.c_int],
        "glTexEnvfv": [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p],
        "glReadPixels": [ctypes.c_int] * 4 + [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p], "glFinish": [],
        "glScissor": [ctypes.c_int] * 4, "glTranslatef": [ctypes.c_float] * 3, "glScalef": [ctypes.c_float] * 3,
        "glVertex2f": [ctypes.c_float] * 2, "glDrawArrays": [ctypes.c_uint, ctypes.c_int, ctypes.c_int],
    }

    def __init__(self):
        self.lib = ctypes.WinDLL("opengl32")
        for name, args in self._sigs.items():
            f = getattr(self.lib, name)
            f.argtypes, f.restype = args, None
            setattr(self, name[2:], f)
        self.wglCreateContext = self.lib.wglCreateContext
        self.wglCreateContext.argtypes, self.wglCreateContext.restype = [ctypes.c_void_p], ctypes.c_void_p
        self.wglMakeCurrent = self.lib.wglMakeCurrent
        self.wglMakeCurrent.argtypes, self.wglMakeCurrent.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int
        self.wglDeleteContext = self.lib.wglDeleteContext
        self.wglDeleteContext.argtypes, self.wglDeleteContext.restype = [ctypes.c_void_p], ctypes.c_int


class PIXELFORMATDESCRIPTOR(ctypes.Structure):
    _fields_ = [("nSize", ctypes.c_ushort), ("nVersion", ctypes.c_ushort), ("dwFlags", ctypes.c_uint),
                ("iPixelType", ctypes.c_ubyte), ("cColorBits", ctypes.c_ubyte)] + \
               [(n, ctypes.c_ubyte) for n in ("cRedBits", "cRedShift", "cGreenBits", "cGreenShift", "cBlueBits",
                                               "cBlueShift", "cAlphaBits", "cAlphaShift", "cAccumBits",
                                               "cAccumRedBits", "cAccumGreenBits", "cAccumBlueBits",
                                               "cAccumAlphaBits", "cDepthBits", "cStencilBits", "cAuxBuffers",
                                               "iLayerType", "bReserved")] + \
               [("dwLayerMask", ctypes.c_uint), ("dwVisibleMask", ctypes.c_uint), ("dwDamageMask", ctypes.c_uint)]
