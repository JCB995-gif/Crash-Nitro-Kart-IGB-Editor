#!/usr/bin/env python3
"""Files of a PS2 disc image (.iso) replaced in place: ASSETS.GOB and ASSETS.GFC of Crash Nitro Kart put back into
its .iso, so that the game (or PCSX2) reads the modded archive.

A PS2 DVD is ISO 9660 (what the console reads: the directory records give each file's first sector and size) with a
UDF 1.02 bridge (what Windows reads: File Entries with the size and the extent, inside a partition). A file that
still fits its sectors is written over them and both get its size; a bigger one goes to the end of the image (its old
sectors stay as they were): its ISO 9660 record and UDF File Entry point there, the volume's size, the UDF
partition's length and the anchor at the last sector follow. Every UDF descriptor changed gets its tag's checksum and
CRC again.

    cnk_iso.py list  <game.iso> [filter]
    cnk_iso.py put   <game.iso> <file> [<file>...]     (each by its name: ASSETS.GOB, ASSETS.GFC)
"""
from __future__ import annotations

from i18n import tr
import struct
import sys
from pathlib import Path

SECTOR = 2048
CHUNK = 1 << 20


class IsoError(ValueError):
    pass


def crc16(data: bytes) -> int:
    """UDF's descriptor CRC (CRC-ITU-T: x^16 + x^12 + x^5 + 1, from 0)."""
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def retag(d: bytearray, location: int | None = None) -> None:
    """A UDF descriptor's tag made right again (its location, its body's CRC, its checksum)."""
    if location is not None:
        struct.pack_into("<I", d, 12, location)
    n = struct.unpack_from("<H", d, 10)[0]
    struct.pack_into("<H", d, 8, crc16(bytes(d[16:16 + n])))
    d[4] = sum(d[k] for k in range(16) if k != 4) & 0xFF


def tag_ok(d: bytes) -> bool:
    n = struct.unpack_from("<H", d, 10)[0]
    return d[4] == sum(d[k] for k in range(16) if k != 4) & 0xFF and \
        struct.unpack_from("<H", d, 8)[0] == crc16(bytes(d[16:16 + n]))


class File:
    def __init__(self, path: str, lba: int, size: int, record_at: int):
        self.path, self.lba, self.size, self.record_at = path, lba, size, record_at
        self.udf_fe: int | None = None  # the sector of its UDF File Entry

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def room(self) -> int:
        return (self.size + SECTOR - 1) // SECTOR * SECTOR


class Iso:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            self.f = f
            self._read()
        self.f = None

    def sector(self, n: int, count: int = 1) -> bytes:
        self.f.seek(n * SECTOR)
        return self.f.read(SECTOR * count)

    def _read(self) -> None:
        pvd = self.sector(16)
        if pvd[:6] != b"\x01CD001":
            raise IsoError("not an ISO 9660 image (no volume descriptor at sector 16)")
        self.volume = struct.unpack_from("<I", pvd, 80)[0]
        self.files: dict[str, File] = {}
        root = pvd[156:156 + 34]
        self._walk(struct.unpack_from("<I", root, 2)[0], struct.unpack_from("<I", root, 10)[0], "/", 0)
        self.f.seek(0, 2)
        self.sectors = self.f.tell() // SECTOR
        self.udf = None
        try:
            self._read_udf()
        except (struct.error, IsoError, IndexError):
            self.udf = None

    def _walk(self, lba: int, size: int, path: str, depth: int) -> None:
        if depth > 16:
            return
        data = self.sector(lba, (size + SECTOR - 1) // SECTOR)
        p = 0
        while p < size:
            n = data[p]
            if n == 0:
                p = (p // SECTOR + 1) * SECTOR
                continue
            rec = data[p:p + n]
            ext, sz, flags, ln = struct.unpack_from("<I", rec, 2)[0], struct.unpack_from("<I", rec, 10)[0], rec[25], \
                rec[32]
            name = rec[33:33 + ln]
            if name not in (b"\0", b"\1"):
                nm = name.decode("latin-1").split(";")[0]
                if flags & 2:
                    self._walk(ext, sz, path + nm + "/", depth + 1)
                else:
                    self.files[(path + nm).upper()] = File(path + nm, ext, sz, lba * SECTOR + p)
            p += n

    def _read_udf(self) -> None:
        a = self.sector(256)
        if struct.unpack_from("<H", a, 0)[0] != 2:
            return
        u = {"vds": [struct.unpack_from("<II", a, 16), struct.unpack_from("<II", a, 24)], "pd": [], "lvid": None}
        for length, loc in u["vds"]:
            for k in range(length // SECTOR):
                d = self.sector(loc + k)
                t = struct.unpack_from("<H", d, 0)[0]
                if t == 5:
                    u["pd"].append(loc + k)
                    u["start"], u["length"] = struct.unpack_from("<II", d, 188)
                elif t == 6 and "fsd" not in u:
                    u["fsd"] = struct.unpack_from("<I", d, 252)[0]
                    lvid_len, lvid_loc = struct.unpack_from("<II", d, 432)
                    u["lvid"] = lvid_loc if lvid_len else None
                elif t == 8:
                    break
        if "start" not in u or "fsd" not in u:
            return
        start = u["start"]
        fsd = self.sector(start + u["fsd"])
        root_lbn = struct.unpack_from("<I", fsd, 404)
        self.udf = u
        self._udf_dir(root_lbn[0], "/", 0)

    def _udf_fe(self, lbn: int) -> tuple[bytes, list[tuple[int, int]], int]:
        d = self.sector(self.udf["start"] + lbn)
        if struct.unpack_from("<H", d, 0)[0] != 261:
            raise IsoError("not a UDF File Entry")
        info = struct.unpack_from("<Q", d, 56)[0]
        lea, lad = struct.unpack_from("<II", d, 168)
        if struct.unpack_from("<H", d, 34)[0] & 7 != 0:  # short allocation descriptors only
            raise IsoError("UDF allocation descriptors of a kind not handled")
        ads = [struct.unpack_from("<II", d, 176 + lea + k) for k in range(0, lad, 8)]
        return d, ads, info

    def _udf_dir(self, lbn: int, path: str, depth: int) -> None:
        if depth > 16:
            return
        _, ads, info = self._udf_fe(lbn)
        data = b"".join(self.sector(self.udf["start"] + pos, ((ln & 0x3FFFFFFF) + SECTOR - 1) // SECTOR)
                        [:ln & 0x3FFFFFFF] for ln, pos in ads)[:info]
        p = 0
        while p + 38 <= len(data):
            if struct.unpack_from("<H", data, p)[0] != 257:
                break
            chars, lfi = data[p + 18], data[p + 19]
            icb = struct.unpack_from("<I", data, p + 24)[0]
            liu = struct.unpack_from("<H", data, p + 36)[0]
            name = data[p + 38 + liu:p + 38 + liu + lfi]
            nm = name[1:].decode("latin-1") if name[:1] == b"\x08" else name[1:].decode("utf-16-be", "replace")
            if lfi and not chars & 8:  # not the parent
                if chars & 2:
                    self._udf_dir(icb, path + nm + "/", depth + 1)
                else:
                    f = self.files.get((path + nm).upper())
                    if f is not None:
                        f.udf_fe = self.udf["start"] + icb
            size = 38 + liu + lfi
            p += size + (-size % 4)

    # Writing.

    def put(self, sources: dict[str, Path], progress=None) -> list[str]:
        """Files written over the image's files of the same name (ASSETS.GOB, ...). What was done."""
        notes = []
        todo = []
        for name, src in sources.items():
            key = next((k for k in self.files if k.rsplit("/", 1)[-1] == name.upper()), None)
            if key is None:
                raise IsoError(f"{name}: not on the disc")
            todo.append((self.files[key], Path(src)))
        with open(self.path, "r+b") as f:
            self.f = f
            try:
                for fi, src in todo:
                    size = src.stat().st_size
                    if size <= fi.room:
                        changed = self._copy(src, fi.lba, size, progress)
                        self._set(fi, fi.lba, size)
                        notes.append(tr('{0}: replaced in place ({1} KB changed)', fi.name, changed // 1024))
                    else:
                        lba = self._grow(size)
                        self._copy(src, lba, size, progress)
                        self._set(fi, lba, size)
                        notes.append(tr('{0}: moved to the end of the image (sector {1})', fi.name, lba))
            finally:
                self.f = None
        self.__init__(self.path)
        for fi, src in todo:  # read back through both directories
            again = self.files[fi.path.upper()]
            if self.read(again) != src.read_bytes():
                raise IsoError(f"{fi.name}: read back different")
        return notes

    def read(self, fi: File, udf: bool = False) -> bytes:
        with open(self.path, "rb") as f:
            if udf and fi.udf_fe is not None:
                self.f = f
                try:
                    _, ads, info = self._udf_fe(fi.udf_fe - self.udf["start"])
                finally:
                    self.f = None
                f.seek((self.udf["start"] + ads[0][1]) * SECTOR)
                return f.read(info)
            f.seek(fi.lba * SECTOR)
            return f.read(fi.size)

    def _copy(self, src: Path, lba: int, size: int, progress) -> int:
        """src over the sectors from lba (its last one padded with zeros), only where it differs; bytes changed."""
        f = self.f
        changed = 0
        with open(src, "rb") as s:
            at = 0
            while at < size:
                new = s.read(CHUNK)
                if len(new) < CHUNK:
                    new += b"\0" * (-len(new) % SECTOR)
                f.seek(lba * SECTOR + at)
                old = f.read(len(new))
                if old != new:
                    f.seek(lba * SECTOR + at)
                    f.write(new)
                    changed += len(new)
                at += CHUNK
                if progress:
                    progress(min(at, size), size)
        return changed

    def _set(self, fi: File, lba: int, size: int) -> None:
        """Its ISO 9660 record and UDF File Entry: where it is now and how big."""
        f = self.f
        f.seek(fi.record_at + 2)
        f.write(struct.pack("<I", lba) + struct.pack(">I", lba) + struct.pack("<I", size) + struct.pack(">I", size))
        if fi.udf_fe is not None and self.udf:
            f.seek(fi.udf_fe * SECTOR)
            d = bytearray(f.read(SECTOR))
            lea, lad = struct.unpack_from("<II", d, 168)
            if lad != 8:
                raise IsoError(f"{fi.name}: a UDF entry of {lad // 8} extents (one expected)")
            struct.pack_into("<Q", d, 56, size)
            struct.pack_into("<Q", d, 64, (size + SECTOR - 1) // SECTOR)  # logical blocks recorded
            struct.pack_into("<II", d, 176 + lea, size, lba - self.udf["start"])
            retag(d)
            f.seek(fi.udf_fe * SECTOR)
            f.write(d)

    def _grow(self, size: int) -> int:
        """Room for size bytes at the image's end: the sector it starts at. The volume's size, the UDF partition and
        the anchor at the last sector made to take it in."""
        f = self.f
        f.seek(0, 2)
        n = f.tell() // SECTOR
        anchor = None
        if self.udf:
            f.seek((n - 1) * SECTOR)
            last = f.read(SECTOR)
            if struct.unpack_from("<H", last, 0)[0] == 2:  # the anchor at the end: the file goes in its place
                anchor = bytearray(last)
                n -= 1
        lba = n
        end = lba + (size + SECTOR - 1) // SECTOR
        f.seek(end * SECTOR - 1)
        f.write(b"\0")
        total = end
        if anchor is not None:
            retag(anchor, end)
            f.seek(end * SECTOR)
            f.write(anchor)
            total = end + 1
        f.seek(16 * SECTOR + 80)
        f.write(struct.pack("<I", total) + struct.pack(">I", total))
        if self.udf:
            length = end - self.udf["start"]
            for loc in self.udf["pd"]:
                f.seek(loc * SECTOR)
                d = bytearray(f.read(SECTOR))
                struct.pack_into("<I", d, 192, length)
                retag(d)
                f.seek(loc * SECTOR)
                f.write(d)
            if self.udf["lvid"] is not None:
                f.seek(self.udf["lvid"] * SECTOR)
                d = bytearray(f.read(SECTOR))
                if struct.unpack_from("<H", d, 0)[0] == 9:
                    parts = struct.unpack_from("<I", d, 72)[0]
                    if parts == 1:
                        struct.pack_into("<I", d, 80 + 4 * parts, length)  # the size table, after the free space
                        retag(d)
                        f.seek(self.udf["lvid"] * SECTOR)
                        f.write(d)
            self.udf["length"] = length
        return lba


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 1
    iso = Iso(argv[2])
    if argv[1] == "list":
        flt = argv[3].upper() if len(argv) > 3 else ""
        for k, fi in sorted(iso.files.items(), key=lambda kv: kv[1].lba):
            if flt in k:
                print(f"{fi.lba:8d} {fi.size:11d}  {fi.path}  {'UDF' if fi.udf_fe else ''}")
    elif argv[1] == "put" and len(argv) > 3:
        for n in iso.put({Path(a).name: Path(a) for a in argv[3:]}):
            print(n)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
