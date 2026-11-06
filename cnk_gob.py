#!/usr/bin/env python3
"""Crash Nitro Kart's archive (PS2, Vicarious Visions 2003): ASSETS.GFC (the index) and ASSETS.GOB (the data), read
and written.

GFC (big-endian):
  0x00  u32 magic 0x00008008
  0x04  u32 size of the .GOB
  0x08  u32 block count  (NB)
  0x0C  u32 file count   (NF)
  0x10  NB x {u32 stored_size, u32 gob_offset, u32 next_block}   next 0x7FFF = end of chain
        NB x u32 checksum: the block's Adler-32, 'STBL' to 'ENBL' included
        NF x {u32 name_hash (the path's CRC-32), u32 unpacked_size, u32 first_block}
        NF x char[72] path (NUL-terminated, rest is junk)
GOB: blocks at 0x800-aligned offsets, the rest of their last sector zeros; each block is 'STBL' + method byte +
  payload + 'ENBL': 'z' a zlib stream (unpacks to <= 64 KiB), '0' the bytes as they are.
A file is its chain of blocks one after another. In the game's archive every file's blocks follow one another in the
table (first, first + 1, ...) in the files' order, and in the .GOB.

Written: a file's new bytes in new blocks, in its own sectors when they fit there (the .GOB keeps its size), else at
the .GOB's end (its old sectors zeroed); the table rebuilt in the same order (the files after it move up or down when
it takes more or fewer blocks). The first time, the originals are kept beside them (ASSETS.GOB.orig, .GFC.orig).

    cnk_gob.py list    ASSETS.GFC [filter]
    cnk_gob.py extract ASSETS.GFC OUTDIR filter [filter ...]   (filters: case-insensitive pieces of the path)
    cnk_gob.py replace ASSETS.GFC path/in/archive new_file
    cnk_gob.py check   ASSETS.GFC                               (every block's checksum)
"""
from __future__ import annotations

from i18n import tr
import shutil
import struct
import sys
import zlib
from pathlib import Path

END = 0x7FFF
SECTOR = 0x800
CHUNK = 0x10000  # a block's bytes unpacked, at most


class GobError(ValueError):
    pass


class Entry:
    def __init__(self, name: str, size: int, first: int, name_hash: int, raw: bytes = b""):
        self.name, self.size, self.first, self.hash, self.raw = name, size, first, name_hash, raw

    @property
    def path(self) -> str:
        """The path with forward slashes and no leading ".\\" (ps2/gfx/chars/tiny.igb)."""
        parts = [p for p in self.name.replace("\\", "/").split("/") if p not in ("", ".")]
        return "/".join(parts)

    def __repr__(self):
        return f"<{self.path} {self.size}>"


def sectors(n: int) -> int:
    """Bytes to the sectors that hold them (in bytes)."""
    return (n + SECTOR - 1) // SECTOR * SECTOR


def encode(data: bytes) -> list[bytes]:
    """A file's bytes as blocks: 64 KiB at a time, packed with zlib (stored as they are when that is no smaller)."""
    out = []
    for k in range(0, max(1, len(data)), CHUNK):
        chunk = data[k:k + CHUNK]
        packed = zlib.compress(chunk, 9)
        out.append(b"STBLz" + packed + b"ENBL" if len(packed) < len(chunk) else b"STBL0" + chunk + b"ENBL")
    return out


def decode(raw: bytes, what: str = "") -> bytes:
    if raw[:4] != b"STBL":
        raise GobError(f"{what}: bad block")
    if raw[4:5] == b"z":
        return zlib.decompress(raw[5:])
    if raw[4:5] == b"0":
        return raw[5:-4] if raw.endswith(b"ENBL") else raw[5:]
    raise GobError(f"{what}: unknown packing {raw[4:5]!r}")


class Gob:
    def __init__(self, gfc_path: Path | str, gob_path: Path | str | None = None):
        gfc_path = Path(gfc_path)
        gfc = gfc_path.read_bytes()
        if gob_path is None:
            for ext in (".GOB", ".gob", ".Gob"):
                if gfc_path.with_suffix(ext).is_file():
                    gob_path = gfc_path.with_suffix(ext)
                    break
            else:
                raise GobError(f"no .GOB beside {gfc_path.name}")
        self.gfc_path, self.gob_path = gfc_path, Path(gob_path)
        if len(gfc) < 16:
            raise GobError("GFC too short")
        magic, self.gob_size, nblk, nfile = struct.unpack_from(">4I", gfc, 0)
        if magic != 0x8008:
            raise GobError(f"not a Crash Nitro Kart .GFC (magic {magic:#x})")
        if 16 + 16 * nblk + 84 * nfile > len(gfc):
            raise GobError("GFC too short for its tables")
        p = 16
        self.blocks = [struct.unpack_from(">3I", gfc, p + 12 * i) for i in range(nblk)]
        p += 12 * nblk
        self.checksums = list(struct.unpack_from(f">{nblk}I", gfc, p))
        p += 4 * nblk
        ents = [struct.unpack_from(">3I", gfc, p + 12 * i) for i in range(nfile)]
        p += 12 * nfile
        self.files: list[Entry] = []
        for i, (h, size, first) in enumerate(ents):
            raw = gfc[p + 72 * i:p + 72 * i + 72]
            self.files.append(Entry(raw.split(b"\0")[0].decode("latin-1"), size, first, h, raw))
        self.tail = gfc[p + 72 * nfile:]
        platforms = {e.path.split("/", 1)[0].lower() for e in self.files} & {"ps2", "xbox"}
        self.platform = next(iter(platforms)) if len(platforms) == 1 else None

    def chain(self, first: int):
        b, seen = first, 0
        while b != END:
            if b >= len(self.blocks) or seen > len(self.blocks):
                raise GobError("broken block chain")
            yield self.blocks[b]
            b = self.blocks[b][2]
            seen += 1

    def chain_of(self, entry: Entry) -> list[int]:
        out, b = [], entry.first
        while b != END:
            if b >= len(self.blocks) or len(out) > len(self.blocks):
                raise GobError(f"{entry.path}: broken block chain")
            out.append(b)
            b = self.blocks[b][2]
        return out

    def read(self, entry: Entry) -> bytes:
        out = bytearray()
        with open(self.gob_path, "rb") as f:
            for sz, off, _ in self.chain(entry.first):
                f.seek(off)
                out += decode(f.read(sz), entry.path)
        if len(out) != entry.size:
            raise GobError(f"{entry.path}: {len(out)} bytes for {entry.size}")
        return bytes(out)

    def match(self, filters: list[str]) -> list[Entry]:
        fl = [x.lower() for x in filters]
        return [e for e in self.files if not fl or any(x in e.path.lower() for x in fl)]

    def entry(self, path: str) -> Entry:
        want = path.replace("\\", "/").strip("./").lower()
        for e in self.files:
            if e.path.lower() == want:
                return e
        raise GobError(f"{path}: not in the archive")

    def igbs(self) -> list[Entry]:
        return sorted((e for e in self.files if e.path.lower().endswith(".igb")), key=lambda e: e.path.lower())

    def check(self) -> list[str]:
        """The blocks whose checksum is not their Adler-32 (none in a good archive)."""
        bad = []
        with open(self.gob_path, "rb") as f:
            for i, ((sz, off, _), ck) in enumerate(zip(self.blocks, self.checksums)):
                f.seek(off)
                if zlib.adler32(f.read(sz)) != ck:
                    bad.append(f"block {i} at {off:#x}")
        return bad

    # Writing.

    def backup(self) -> list[Path]:
        """The originals kept once, beside them (ASSETS.GOB.orig, ASSETS.GFC.orig)."""
        made = []
        for p in (self.gob_path, self.gfc_path):
            orig = p.with_name(p.name + ".orig")
            if not orig.exists():
                shutil.copyfile(p, orig)
                made.append(orig)
        return made

    def has_backup(self) -> bool:
        return all(p.with_name(p.name + ".orig").is_file() for p in (self.gob_path, self.gfc_path))

    def restore(self) -> None:
        """The originals back in place (from the .orig copies)."""
        for p in (self.gob_path, self.gfc_path):
            orig = p.with_name(p.name + ".orig")
            if not orig.is_file():
                raise GobError(f"no {orig.name} to restore from")
            shutil.copyfile(orig, p)
        self.__init__(self.gfc_path, self.gob_path)

    def replace(self, changes: dict[str, bytes], backup: bool = True) -> list[str]:
        """Files' new bytes written into the .GOB and the .GFC (paths as entry.path gives them). What was done."""
        targets = {}
        for path, data in changes.items():
            e = self.entry(path)
            targets[self.files.index(e)] = bytes(data)
        # The table as the game's: every file's blocks in order, the files one after another.
        at = 0
        for e in self.files:
            ch = self.chain_of(e)
            if ch != list(range(at, at + len(ch))):
                raise GobError(f"{e.path}: its blocks are not where the game's archive has them; not written")
            at += len(ch)
        if at != len(self.blocks):
            raise GobError("blocks that belong to no file; not written")
        if backup:
            self.backup()
        notes = []
        writes: list[tuple[int, bytes]] = []  # (offset, bytes) into the .GOB
        end = sectors(self.gob_size)
        new_blocks: list[tuple[int, int, int]] = []
        new_sums: list[int] = []
        new_firsts: list[int] = []
        new_sizes: list[int] = []
        for k, e in enumerate(self.files):
            ch = self.chain_of(e)
            new_firsts.append(len(new_blocks))
            if k not in targets:
                for j, b in enumerate(ch):
                    sz, off, _ = self.blocks[b]
                    new_blocks.append((sz, off, END if j == len(ch) - 1 else len(new_blocks) + 1))
                    new_sums.append(self.checksums[b])
                new_sizes.append(e.size)
                continue
            data = targets[k]
            raws = encode(data)
            start = self.blocks[ch[0]][1]
            room = sum(sectors(self.blocks[b][0]) for b in ch)
            need = sum(sectors(len(r)) for r in raws)
            if need <= room:
                pos = start
                writes.append((start, b"\0" * room))  # its old sectors cleared, then the new blocks over them
                where = tr("in its original sectors")
            else:
                pos = end
                writes.append((start, b"\0" * room))
                end += need
                where = tr("at the end of the GOB")
            for j, r in enumerate(raws):
                writes.append((pos, r))
                new_blocks.append((len(r), pos, END if j == len(raws) - 1 else len(new_blocks) + 1))
                new_sums.append(zlib.adler32(r))
                pos += sectors(len(r))
            new_sizes.append(len(data))
            notes.append(tr('{0}: {1} bytes in {2} blocks, {3}', e.path, len(data), len(raws), where))
        gob_size = end if end > sectors(self.gob_size) else self.gob_size
        # The .GOB: the blocks written (it grows when one went to its end), then the .GFC.
        with open(self.gob_path, "r+b") as f:
            for off, raw in writes:
                f.seek(off)
                f.write(raw)
            if gob_size > self.gob_size:
                f.seek(0, 2)
                have = f.tell()
                if have < gob_size:
                    f.write(b"\0" * (gob_size - have))
                f.truncate(gob_size)
        out = bytearray(struct.pack(">4I", 0x8008, gob_size, len(new_blocks), len(self.files)))
        out += b"".join(struct.pack(">3I", *b) for b in new_blocks)
        out += struct.pack(f">{len(new_sums)}I", *new_sums)
        out += b"".join(struct.pack(">3I", e.hash, size, first)
                        for e, size, first in zip(self.files, new_sizes, new_firsts))
        out += b"".join(e.raw for e in self.files)
        out += self.tail
        tmp = self.gfc_path.with_name(self.gfc_path.name + ".tmp")
        tmp.write_bytes(bytes(out))
        tmp.replace(self.gfc_path)
        self.__init__(self.gfc_path, self.gob_path)
        # Read back: what was written is what was given.
        for path, data in changes.items():
            if self.read(self.entry(path)) != data:
                raise GobError(f"{path}: read back different")
        return notes


def name_hash(name: str) -> int:
    """A path's hash in the .GFC (as it is written there: «.\\ps2\\gfx\\chars\\tiny.igb»)."""
    return zlib.crc32(name.encode("latin-1"))


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 1
    cmd, gfc = argv[1], argv[2]
    g = Gob(gfc)
    if cmd == "list":
        sel = g.match(argv[3:])
        for e in sel:
            print("%10d  %s" % (e.size, e.path))
        print("%d of %d files" % (len(sel), len(g.files)))
    elif cmd == "extract":
        outdir, filters = Path(argv[3]), argv[4:]
        if not filters:
            print('give at least one filter ("" for everything)')
            return 1
        for e in g.match(filters):
            dst = outdir.joinpath(*e.path.split("/"))
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(g.read(e))
            print("%10d  %s" % (e.size, dst))
    elif cmd == "replace" and len(argv) > 4:
        for n in g.replace({argv[3]: Path(argv[4]).read_bytes()}):
            print(n)
    elif cmd == "check":
        bad = g.check()
        print("\n".join(bad) or f"{len(g.blocks)} blocks, all good")
        return 1 if bad else 0
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
