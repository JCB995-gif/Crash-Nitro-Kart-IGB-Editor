"""Browse every serialized object and memory block without expanding cyclic references."""
from i18n import tr
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import igb_file


class Inspector(ttk.Frame):
    PAGE = 250
    BYTES = 1024

    def __init__(self, parent):
        super().__init__(parent, padding=6)
        self.file = None
        self.matches = []
        self.page = 0
        self.memory_offset = 0
        self.selected = None
        self.targets = {}
        self.summary = ttk.Label(self, wraplength=330, justify="left")
        self.summary.pack(fill="x")
        self.kind = tk.StringVar(value=tr('All objects'))
        self.classes = ttk.Combobox(self, textvariable=self.kind, state="readonly")
        self.classes.pack(fill="x", pady=4)
        self.classes.bind("<<ComboboxSelected>>", lambda e: self.filter())
        self.query = tk.StringVar()
        ttk.Entry(self, textvariable=self.query).pack(fill="x")
        self.query.trace_add("write", lambda *a: self.filter())
        self.objects = tk.Listbox(self, height=7, exportselection=False)
        self.objects.pack(fill="both", expand=True, pady=4)
        self.objects.bind("<<ListboxSelect>>", self.pick)
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Button(row, text="◀", width=3, command=lambda: self.turn(-1)).pack(side="left")
        ttk.Button(row, text="▶", width=3, command=lambda: self.turn(1)).pack(side="left")
        self.pages = ttk.Label(row)
        self.pages.pack(side="left", padx=4)
        self.fields = ttk.Treeview(self, columns=("type", "value"), show="tree headings", height=6)
        self.fields.heading("#0", text='Field / slot')
        self.fields.heading("type", text='Type')
        self.fields.heading("value", text='Value / reference')
        self.fields.column("#0", width=65, stretch=False)
        self.fields.column("type", width=90, stretch=False)
        self.fields.column("value", width=150)
        self.fields.pack(fill="both", expand=True, pady=4)
        self.fields.bind("<Double-Button-1>", self.follow)
        self.fields.bind("<Return>", self.follow)
        self.fields.bind("<<TreeviewSelect>>", self.field_details)
        self.details = tk.Text(self, height=9, width=40, wrap="none", state="disabled", font=("Consolas", 9))
        self.details.pack(fill="both", expand=True)
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="−1 KB", command=lambda: self.memory_page(-1)).pack(side="left")
        ttk.Button(row, text="+1 KB", command=lambda: self.memory_page(1)).pack(side="left")
        ttk.Button(row, text='Export block…', command=self.export_memory).pack(side="left", padx=4)

    def set_file(self, f, notes=()):
        self.file = f
        self.selected = None
        self.fields.delete(*self.fields.get_children())
        self.text("\n".join(notes))
        if f is None:
            self.summary.configure(text='Open an IGB to inspect every object and data block.')
            self.classes.configure(values=(tr('All objects'),))
        else:
            used = f.classes_used()
            self.summary.configure(text=tr('IGB v{0} · {1} objects and blocks · {2} classes\nDouble-click a reference to follow it.', f.version, len(f.objects), len(used)))
            self.classes.configure(values=[tr('All objects'), tr('Memory')] + sorted(used))
        self.kind.set(tr('All objects'))
        self.query.set("")
        self.filter()

    def filter(self):
        kind, query = self.kind.get(), self.query.get().casefold().strip()
        self.matches = []
        for o in self.file.objects if self.file else []:
            is_mem = isinstance(o, igb_file.Mem)
            if kind != tr('All objects') and not (kind == tr('Memory') and is_mem or not is_mem and o.name == kind):
                continue
            label = tr('{0}: memory ({1} bytes)', o.index, len(o.data)) if is_mem else f"{o.index}: {o.name}"
            names = "" if is_mem else " ".join(v for v in o.slots.values() if isinstance(v, str))
            if query and query not in (label + " " + names).casefold():
                continue
            self.matches.append((o.index, label + (" — " + names[:100] if names else "")))
        self.page = 0
        self.populate()

    def populate(self):
        self.objects.delete(0, "end")
        for _, label in self.matches[self.page:self.page + self.PAGE]:
            self.objects.tk.call(self.objects._w, "insert", "end", label)
        self.pages.configure(text=f"{self.page + 1 if self.matches else 0}–"
                             f"{min(self.page + self.PAGE, len(self.matches))} / {len(self.matches)}")

    def turn(self, direction):
        last = max(0, (len(self.matches) - 1) // self.PAGE * self.PAGE)
        self.page = min(last, max(0, self.page + direction * self.PAGE))
        self.populate()

    def pick(self, _e=None):
        selection = self.objects.curselection()
        if selection:
            self.show_object(self.matches[self.page + selection[0]][0])

    def show_object(self, index):
        if self.file is None or not 0 <= index < len(self.file.objects):
            return
        self.selected = self.file.objects[index]
        self.memory_offset = 0
        self.targets = {}
        self.fields.delete(*self.fields.get_children())
        o = self.selected
        if isinstance(o, igb_file.Mem):
            self.show_memory()
            return
        self.text(tr('#{0} {1}\nParents: {2}', o.index, o.name, " → ".join(n.decode("latin-1") for n in o.klass.chain())))
        for k, (kind, slot, _) in enumerate(o.klass.fields):
            value = o.slots.get(slot)
            target = self.file.ref(o, slot) if kind in (b"ObjectRef", b"MemoryRef") else None
            label = repr(target) if target is not None else repr(value)
            iid = self.fields.insert("", "end", text=f"{k} / {slot}", values=(kind.decode("latin-1"), label[:200]))
            self.targets[iid] = (slot, target.index if target is not None else None)

    def follow(self, _e=None):
        selection = self.fields.selection()
        if selection:
            _, target = self.targets[selection[0]]
            if target is not None:
                self.show_object(target)

    def field_details(self, _e=None):
        selection = self.fields.selection()
        if selection and isinstance(self.selected, igb_file.Obj):
            slot, _ = self.targets[selection[0]]
            self.text(f"#{self.selected.index} {self.selected.name}, slot {slot}\n"
                      + repr(self.selected.slots.get(slot)))

    def text(self, value):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("end", value)
        self.details.configure(state="disabled")

    def show_memory(self):
        o = self.selected
        if not isinstance(o, igb_file.Mem):
            return
        start = self.memory_offset
        rows = [tr('#{0}: {1} bytes; offset 0x{2:X}', o.index, len(o.data), start)]
        for offset in range(start, min(start + self.BYTES, len(o.data)), 16):
            chunk = o.data[offset:offset + 16]
            rows.append(f"{offset:08X}  {chunk.hex(' '):47}  "
                        + "".join(chr(b) if 32 <= b < 127 else "." for b in chunk))
        self.text("\n".join(rows))

    def memory_page(self, direction):
        if isinstance(self.selected, igb_file.Mem):
            last = max(0, (len(self.selected.data) - 1) // self.BYTES * self.BYTES)
            self.memory_offset = min(last, max(0, self.memory_offset + direction * self.BYTES))
            self.show_memory()

    def export_memory(self):
        if not isinstance(self.selected, igb_file.Mem):
            return
        path = filedialog.asksaveasfilename(parent=self, initialfile=f"memory-{self.selected.index}.bin",
                                          defaultextension=".bin")
        if path:
            try:
                from pathlib import Path
                Path(path).write_bytes(self.selected.data)
            except OSError as ex:
                messagebox.showerror('Export', str(ex), parent=self)
