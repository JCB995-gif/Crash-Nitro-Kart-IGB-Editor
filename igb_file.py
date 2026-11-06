#!/usr/bin/env python3
"""An Alchemy .igb (versions 4 to 8; Crash Nitro Kart's are 5) read into its objects and written back.

Each object is an instance of a class with fields by slot (numbers, vectors, strings, references to other objects or
to memory blocks); each memory block is its bytes. Written back, the file is the one read, byte for byte, but for
what was changed: a memory block's bytes (any size), an object's fixed-size field, memory blocks added at the end.

The layout, after the 0x30-byte header (12 words: the directory's size and count, the classes' size and count, the
objects' size and count, the memory's size and count, the field types' size and count, 0xFADA, version and flags):
field types, an alignment table, classes, (two optional tables), the directory (an entry an object or memory block:
its class and fields), (from version 6 an index of the directory), (an info list's index), the objects' fields in
directory order, the memory blocks' bytes in directory order (each padded to 4 bytes).

    igb_file.py <file.igb> [class...]      the classes counted, then the objects of the classes named, field by field
"""
from __future__ import annotations

import struct
import sys
from collections import Counter
from pathlib import Path

FORMATS = {b"Bool": "B", b"Char": "b", b"UnsignedChar": "B", b"Short": "h", b"UnsignedShort": "H", b"Int": "i",
           b"UnsignedInt": "I", b"Long": "q", b"UnsignedLong": "Q", b"Float": "f", b"Double": "d", b"Vec2f": "2f",
           b"Vec3f": "3f", b"Vec4f": "4f", b"Quaternionf": "4f", b"Matrix44f": "16f", b"Enum": "i",
           b"Handle": "I", b"BitField": "I", b"ObjectRef": "i", b"MemoryRef": "i"}
ARRAYS = {b"CharArray": "b", b"UnsignedCharArray": "B", b"FloatArray": "f", b"IntArray": "i",
          b"UnsignedIntArray": "I", b"ShortArray": "h", b"UnsignedShortArray": "H"}
SIZED = {1: "B", 2: "h", 4: "i"}
MAGIC = 0xFADA


class IgbError(ValueError):
    pass


class Klass:
    def __init__(self, name: bytes, fields: list, parent: int):
        self.name, self.fields, self.parent_index, self.parent = name, fields, parent, None

    def chain(self) -> list[bytes]:
        k, out = self, []
        while k:
            out.append(k.name)
            k = k.parent
        return out


class Obj:
    def __init__(self, index: int, klass: Klass):
        self.index, self.klass = index, klass
        self.slots: dict[int, object] = {}
        self.kinds: dict[int, bytes] = {}
        self.at: dict[int, tuple[int, int]] = {}  # slot -> (offset in the objects' section, size) of fixed fields

    @property
    def name(self) -> str:
        return self.klass.name.decode("latin-1")

    def is_a(self, name: str) -> bool:
        return name.encode() in self.klass.chain()

    def __repr__(self):
        return f"<{self.index} {self.name}>"


class Mem:
    def __init__(self, index: int, size: int, kind: int):
        self.index, self.size, self.kind, self.data = index, size, kind, b""
        self.entry = -1  # its directory entry
        self.pad = b""

    def __repr__(self):
        return f"<{self.index} mem {self.size}>"


class External(Obj):
    """An external reference lives in the directory, without an object payload."""

    def __repr__(self):
        return f"<{self.index} external {self.name}>"


class IGB:
    def __init__(self, data: bytes):
        self.d = d = bytes(data)
        if len(d) < 0x30:
            raise IgbError("file too short for an .igb")
        if struct.unpack_from("<I", d, 0x28)[0] == MAGIC:
            e = "<"
        elif struct.unpack_from(">I", d, 0x28)[0] == MAGIC:
            e = ">"
        else:
            raise IgbError("not an .igb (no 0xFADA)")
        self.e = e
        try:
            self._read()
        except (struct.error, IndexError, KeyError) as ex:
            raise IgbError(f"damaged .igb ({ex})") from None

    def _read(self) -> None:
        d, e = self.d, self.e
        self.header = list(struct.unpack_from(e + "12I", d, 0))
        ent_size, ent_count, mo_size, mo_count, ob_size, _, mr_size, _, mf_size, mf_count, _, vf = self.header
        self.version, self.flags = vf & 0xFFFF, vf
        if not 4 <= self.version <= 8:
            raise IgbError(f"version {self.version} not handled (4 to 8 are)")
        pos = 0x30
        if self.version >= 8:
            pos += struct.unpack_from(e + "I", d, pos)[0]
        names, dyn = [], pos + 12 * mf_count
        for i in range(mf_count):
            n = struct.unpack_from(e + "I", d, pos + 12 * i)[0]
            raw = d[dyn:dyn + n].split(b"\0")[0]
            dyn += n
            short = raw[2:] if raw.startswith(b"ig") else raw
            names.append(short[:-9] if short.endswith(b"MetaField") else short)
        self.ftypes = names
        pos += mf_size
        pos += struct.unpack_from(e + "I", d, pos)[0]  # the alignment table
        recs = [struct.unpack_from(e + "IIIIiI", d, pos + 24 * i) for i in range(mo_count)]
        dyn = pos + 24 * mo_count
        self.classes: list[Klass] = []
        for n, _, _, nf, parent, _ in recs:
            nm = d[dyn:dyn + n].split(b"\0")[0]
            dyn += n
            fields = []
            for _ in range(nf):
                t, slot, size = struct.unpack_from(e + "HHH", d, dyn)
                dyn += 6
                fields.append((names[t] if t < len(names) else b"?", slot, size))
            self.classes.append(Klass(nm, fields, parent))
        for k in self.classes:
            if 0 <= k.parent_index < len(self.classes):
                k.parent = self.classes[k.parent_index]
        pos += mo_size
        if vf & 0x40000000:
            pos += struct.unpack_from(e + "I", d, pos)[0]
        if vf & 0x10000000:
            pos += struct.unpack_from(e + "I", d, pos)[0]
        # The directory: an object or a memory block an entry.
        self.dir_at = pos
        self.entries: list[tuple[int, int, list]] = []  # (offset, size, values)
        self.entry_kind: list[bytes] = []
        p = pos
        for _ in range(ent_count):
            t, size = struct.unpack_from(e + "II", d, p)
            vals, q = [], p + 8  # in field order: the slots' numbers differ between versions
            for ft, slot, fsize in self.classes[t].fields:
                used = fsize
                if ft == b"String":
                    n = struct.unpack_from(e + "i", d, q)[0]
                    if n < 0 or q + fsize + n > p + size:
                        raise IgbError("invalid directory string length")
                    vals.append(d[q + 4:q + 4 + n].split(b"\0")[0].decode("latin-1"))
                    used += n
                else:
                    vals.append(struct.unpack_from(e + SIZED[fsize], d, q)[0])
                q += (used + 3) & ~3
            self.entries.append((p, size, vals))
            self.entry_kind.append(self.classes[t].name)
            p += size
        pos += ent_size
        self.index_at = pos
        index = list(range(ent_count))
        if self.version >= 6:
            size, n = struct.unpack_from(e + "II", d, pos)
            index = list(struct.unpack_from(e + f"{n}H", d, pos + 8))
            pos += size
        self.index = index
        self.objects: list[Obj | Mem] = []
        for i in index:
            _, _, vals = self.entries[i]
            if self.entry_kind[i] == b"igMemoryDirEntry":  # name, size, type, ref counted, pool
                m = Mem(len(self.objects), vals[1], vals[2])
                m.entry = i
                self.objects.append(m)
            elif self.entry_kind[i].startswith(b"igExternal"):
                t = struct.unpack_from(e + "I", d, self.entries[i][0])[0]
                o = External(len(self.objects), self.classes[t])
                o.slots = {slot: value for (_, slot, _), value in zip(o.klass.fields, vals)}
                o.kinds = {slot: ft for ft, slot, _ in o.klass.fields}
                self.objects.append(o)
            else:  # name, type
                self.objects.append(Obj(len(self.objects), self.classes[vals[1]]))
        self.info_at = None
        self.info = None
        if self.version >= 5 and self.flags & 0x80000000:  # v4 starts directly with the first object
            self.info_at = pos
            self.info = struct.unpack_from(self.e + "I", d, pos)[0]
            pos += 4
        self.obj_at = pos
        p = pos
        for o in self.objects:
            if not isinstance(o, Obj) or isinstance(o, External):
                continue
            t, size = struct.unpack_from(e + "II", d, p)
            o.klass = self.classes[t]
            q, end = p + 8, p + size
            for ft, slot, fsize in o.klass.fields:
                if q + fsize > end:
                    break
                used = fsize
                fixed = True
                if ft == b"String":
                    n = struct.unpack_from(e + "i", d, q)[0]
                    val = d[q + 4:q + 4 + n].split(b"\0")[0].decode("latin-1")
                    used = fsize + n
                    fixed = False
                elif ft in ARRAYS:
                    c = ARRAYS[ft]
                    val = struct.unpack_from(e + f"{fsize // struct.calcsize(c)}{c}", d, q)
                elif ft in FORMATS:
                    val = struct.unpack_from(e + FORMATS[ft], d, q)
                    val = val[0] if len(val) == 1 else val
                else:
                    val = struct.unpack_from(e + SIZED[fsize], d, q)[0] if fsize in SIZED else d[q:q + fsize]
                o.slots[slot] = val
                o.kinds[slot] = ft
                if fixed:
                    o.at[slot] = (q - pos, fsize)
                q += (used + 3) & ~3
            p = end
        self.objects_raw = bytearray(d[pos:pos + ob_size])
        pos += ob_size
        self.mem_at = pos
        p = pos
        for o in self.objects:
            if isinstance(o, Mem):
                o.data = d[p:p + o.size]
                o.pad = d[p + o.size:p + ((o.size + 3) & ~3)]  # kept as it was (the exporter left 0xCD there)
                p += (o.size + 3) & ~3
        pos += mr_size
        self.tail = d[pos:]
        self.added: list[Mem] = []

    # Reading.

    def ref(self, o, slot: int):
        """The object or memory block a reference field points at (None for none)."""
        if o is None:
            return None
        v = o.slots.get(slot, -1)
        return self.objects[v] if isinstance(v, int) and 0 <= v < len(self.objects) else None

    # A class's fields by their place in it: some files number a class's slots differently (an igImage's width is
    # slot 2 in most of Crash Nitro Kart's, slot 3 in a few), the order and kinds of its fields stay the same.

    def get(self, o, k: int, default=None):
        """An object's k-th field."""
        fl = o.klass.fields if isinstance(o, Obj) else ()
        return o.slots.get(fl[k][1], default) if k < len(fl) else default

    def gref(self, o, k: int):
        """What an object's k-th field (a reference) points at."""
        if not isinstance(o, Obj) or k >= len(o.klass.fields):
            return None
        return self.ref(o, o.klass.fields[k][1])

    def put(self, o: Obj, k: int, value) -> None:
        """An object's k-th field set (fixed size)."""
        self.set_field(o, o.klass.fields[k][1], value)

    def refs(self, o) -> list[tuple[int, object]]:
        """Every (slot, target) an object refers to."""
        return [(s, self.objects[v]) for s, v in o.slots.items()
                if o.kinds[s] in (b"ObjectRef", b"MemoryRef") and isinstance(v, int) and 0 <= v < len(self.objects)]

    def items(self, lst) -> list:
        """An igObjectList's (or an igDataList's of refs) objects."""
        if lst is None or not isinstance(lst, Obj):
            return []
        count = next((v for s, v in sorted(lst.slots.items()) if lst.kinds[s] == b"Int"), 0)
        mem = next((self.objects[v] for s, v in sorted(lst.slots.items()) if lst.kinds[s] == b"MemoryRef"
                    and isinstance(v, int) and 0 <= v < len(self.objects)), None)
        if not isinstance(mem, Mem):
            return []
        out = []
        for k in range(min(count, len(mem.data) // 4)):
            v = struct.unpack_from(self.e + "i", mem.data, 4 * k)[0]
            out.append(self.objects[v] if 0 <= v < len(self.objects) else None)
        return out

    def data_list(self, lst, fmt: str) -> list[tuple]:
        """An igDataList's items (its count at its first Int, its data at its MemoryRef) as struct tuples."""
        if lst is None or not isinstance(lst, Obj):
            return []
        n = next((v for s, v in sorted(lst.slots.items()) if lst.kinds[s] == b"Int"), 0)
        mem = next((self.objects[v] for s, v in sorted(lst.slots.items()) if lst.kinds[s] == b"MemoryRef"
                    and isinstance(v, int) and 0 <= v < len(self.objects)), None)
        if not isinstance(mem, Mem) or not n:
            return []
        size = struct.calcsize(fmt)
        return [struct.unpack_from(self.e + fmt, mem.data, size * k) for k in range(min(n, len(mem.data) // size))]

    def of(self, name: str) -> list[Obj]:
        return [o for o in self.objects if isinstance(o, Obj) and o.is_a(name)]

    def classes_used(self) -> Counter:
        return Counter(o.name for o in self.objects if isinstance(o, Obj))

    # Changing.

    def set_field(self, o: Obj, slot: int, value) -> None:
        """A fixed-size field set (a number, a vector, a reference)."""
        if slot not in o.at:
            raise IgbError(f"{o!r} has no fixed field {slot}")
        off, size = o.at[slot]
        ft = o.kinds[slot]
        fmt = FORMATS.get(ft) or (f"{size // struct.calcsize(ARRAYS[ft])}{ARRAYS[ft]}" if ft in ARRAYS else
                                  SIZED.get(size))
        if fmt is None:
            raise IgbError(f"{o!r} field {slot} ({ft.decode()}) cannot be set")
        vals = value if isinstance(value, (tuple, list)) else (value,)
        raw = struct.pack(self.e + fmt, *vals)
        if len(raw) != size:
            raise IgbError(f"{o!r} field {slot}: {len(raw)} bytes for {size}")
        self.objects_raw[off:off + size] = raw
        o.slots[slot] = value if isinstance(value, (tuple, list)) or len(vals) != 1 else vals[0]
        if isinstance(o.slots[slot], list):
            o.slots[slot] = tuple(o.slots[slot])

    def set_list(self, lst: Obj, data: bytes, count: int) -> Mem:
        """An igDataList's items replaced: its memory block's bytes and its count (and capacity, when it has one:
        the Int fields after the count)."""
        ints = [s for s in sorted(lst.slots) if lst.kinds[s] == b"Int"]
        mem = self.ref(lst, next(s for s in sorted(lst.slots) if lst.kinds[s] == b"MemoryRef"))
        if not isinstance(mem, Mem):
            raise IgbError(f"{lst!r} has no memory block")
        mem.data, mem.size = bytes(data), len(data)
        for k, s in enumerate(ints[:2]):
            if k == 0 or lst.slots[s] >= lst.slots[ints[0]]:  # the capacity follows the count when it had it
                self.set_field(lst, s, count)
        return mem

    def new_memory(self, data: bytes, like: Mem) -> Mem:
        """A memory block added (its directory entry a copy of like's): its index for the reference fields."""
        if self.version >= 6:
            raise IgbError("adding memory blocks to version 6+ files is not handled")
        m = Mem(len(self.objects), len(data), like.kind)
        m.data = bytes(data)
        m.entry = -1 - len(self.added)
        m.like = like
        self.objects.append(m)
        self.added.append(m)
        return m

    # Writing.

    def write(self) -> bytes:
        e, d = self.e, self.d
        ent_size, ent_count = self.header[0], self.header[1]
        out = bytearray(d[:self.dir_at])
        size_slot = None
        # The directory, the memory blocks' sizes as they are now, then the entries added.
        dir_bytes = bytearray(d[self.dir_at:self.dir_at + ent_size])
        mem_of_entry = {o.entry: o for o in self.objects if isinstance(o, Mem) and o.entry >= 0}
        for i, (p, size, vals) in enumerate(self.entries):
            m = mem_of_entry.get(i)
            if m is None:
                continue
            fields = self.classes[struct.unpack_from(e + "I", d, p)[0]].fields
            q = p + 8 + ((fields[0][2] + 3) & ~3)  # the size is the second field
            size_slot = (fields[0][2] + 3) & ~3
            struct.pack_into(e + SIZED[fields[1][2]], dir_bytes, q - self.dir_at, m.size)
        for m in self.added:
            p, size, _ = self.entries[m.like.entry]
            raw = bytearray(d[p:p + size])
            struct.pack_into(e + "i", raw, 8 + size_slot, m.size)
            dir_bytes += raw
        out += dir_bytes
        if self.version >= 6:  # (no entries added there)
            size = struct.unpack_from(e + "I", d, self.index_at)[0]
            out += d[self.index_at:self.index_at + size]
        if self.info_at is not None:
            out += d[self.info_at:self.info_at + 4]
        out += self.objects_raw
        mem = bytearray()
        for o in self.objects:
            if isinstance(o, Mem):
                mem += o.data
                mem += o.pad if len(o.pad) == -len(o.data) % 4 else b"\0" * (-len(o.data) % 4)
        out += mem
        out += self.tail
        h = list(self.header)
        h[0] = len(dir_bytes)
        h[1] = ent_count + len(self.added)
        h[6] = len(mem)
        h[7] = sum(1 for o in self.objects if isinstance(o, Mem))
        struct.pack_into(e + "12I", out, 0, *h)
        return bytes(out)


def show(f: IGB, o, limit: int = 8) -> str:
    out = []
    for s in sorted(o.slots):
        v, k = o.slots[s], o.kinds[s].decode()
        if k in ("ObjectRef", "MemoryRef") and isinstance(v, int) and 0 <= v < len(f.objects):
            v = f.objects[v]
        elif isinstance(v, tuple) and len(v) > limit:
            v = f"{v[:limit]}... ({len(v)})"
        elif isinstance(v, float):
            v = round(v, 4)
        elif isinstance(v, tuple):
            v = tuple(round(x, 4) if isinstance(x, float) else x for x in v)
        out.append(f"{s}:{k}={v}")
    return " ".join(out)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    f = IGB(Path(sys.argv[1]).read_bytes())
    print("version", f.version, "objects", len(f.objects), "info", f.info)
    for k, n in sorted(f.classes_used().items()):
        print(f"{n:5d} {k}")
    for name in sys.argv[2:]:
        for o in f.of(name):
            print(o, show(f, o))
    return 0


if __name__ == "__main__":
    sys.exit(main())
