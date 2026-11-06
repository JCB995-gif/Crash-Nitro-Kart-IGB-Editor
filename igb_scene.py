"""Alchemy transform-source animations: rigid objects, scene helpers and camera paths."""
from i18n import tr
import math
import igb_file
import igb_model as model


class SceneClip(model.Clip):
    def __init__(self, motion, phase, name):
        super().__init__(name, motion.lengths[phase])
        self.motion, self.scene_phase = motion, phase

    def pose(self, t):
        return []  # Node transforms are independent of a skeleton.


class Motion:
    def __init__(self, f):
        self.f = f
        self.nodes = {o.index: o for o in f.of("igTransform")}
        self.channels = {}
        self.rest = {i: list(f.get(o, model.T_MATRIX, model.IDENTITY)) for i, o in self.nodes.items()}
        self.static_phases = {}
        self.lengths = []
        self._paths = {}
        self.notes = []
        for index, node in self.nodes.items():
            source = f.gref(node, 6)
            if not isinstance(source, igb_file.Obj):
                continue
            sequences = f.items(f.gref(source, 0)) if source.is_a("vvTransformSource") else [source]
            matrices = f.data_list(f.gref(source, 1), "16f") if source.is_a("vvTransformSource") else []
            self.static_phases[index] = matrices
            for phase, sequence in enumerate(sequences):
                if not isinstance(sequence, igb_file.Obj) or not sequence.is_a("igTransformSequence"):
                    continue
                times = [v[0] / 1e9 for v in f.data_list(f.gref(sequence, model.SQ_TIMES), "q")]
                translations = f.data_list(f.gref(sequence, model.SQ_XLATE), "3f")
                rotations = f.data_list(f.gref(sequence, model.SQ_QUAT), "4f")
                scales = f.data_list(f.gref(sequence, model.SQ_SCALE), "3f")
                if any(not all(math.isfinite(v) for v in q) for q in rotations):
                    # Exporting a zero-scale matrix can leave its quaternion NaN.
                    # Its orientation is undefined; carry a valid neighbouring key
                    # so interpolation into that collapsed frame stays finite.
                    valid = [q for q in rotations if all(math.isfinite(v) for v in q)]
                    previous = valid[0] if valid else (0, 0, 0, 1)
                    fixed = []
                    for q in rotations:
                        if all(math.isfinite(v) for v in q):
                            previous = q
                        fixed.append(previous)
                    rotations = fixed
                    self.notes.append(tr('{0}, phase {1}: undefined rotation in the file; a nearby valid key is used for preview.', f.get(node, 0, index), phase + 1))
                length = max(f.get(sequence, 15, 0) / 1e9, times[-1] if times else 0, .001)
                while len(self.lengths) <= phase:
                    self.lengths.append(.001)
                self.lengths[phase] = max(self.lengths[phase], length)
                self.channels[index, phase] = times, translations, rotations, scales

    def local(self, index, phase, t):
        base = self.rest[index]
        channel = self.channels.get((index, phase))
        if channel is None:
            matrices = self.static_phases.get(index, [])
            return list(matrices[phase]) if phase < len(matrices) else base
        times, translations, rotations, scales = channel
        position = model.IgbClip._at(times, translations, t) or base[12:15]
        quat = model.IgbClip._at(times, rotations, t)
        rows = model.quat_matrix(quat) if quat is not None else [base[0:3], base[4:7], base[8:11]]
        scale = model.IgbClip._at(times, scales, t)
        if scale is not None:
            rows = [[v * scale[i] for v in row] for i, row in enumerate(rows)]
        return [*rows[0], 0, *rows[1], 0, *rows[2], 0, *position, 1]

    def world(self, path, phase, t):
        matrix = model.IDENTITY
        for index in path:
            matrix = model.mul(self.local(index, phase, t), matrix)
        return matrix

    def path(self, index, phase):
        key = index, phase
        if key not in self._paths:
            length = self.lengths[phase] if phase < len(self.lengths) else 0
            count = min(240, max(2, round(length * 30)))
            self._paths[key] = [tuple(self.local(index, phase, length * k / (count - 1))[12:15])
                                for k in range(count)]
        return self._paths[key]

    def clips(self):
        return [SceneClip(self, phase, tr('Scene {0}', phase + 1)) for phase in range(len(self.lengths))]
