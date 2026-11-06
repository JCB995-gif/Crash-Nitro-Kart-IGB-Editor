"""English source texts with optional lang/<code>.json translation catalogs.

Catalogs map English messages to translations and declare their name in _name.
Dynamic messages use str.format placeholders. Data values are never translated.
English needs no catalog.
"""
from __future__ import annotations

import json
from pathlib import Path
from string import Formatter
import tkinter as tk
from tkinter import commondialog, messagebox, ttk

SETTINGS = Path.home() / ".cnk-igb-editor.json"
LANG_DIR = Path(__file__).resolve().parent / "lang"
DEFAULT = "en"
_code: str | None = None
_table: dict[str, str] = {}


def _fields(text: str):
    return sorted((field, spec, conversion) for _, field, spec, conversion in Formatter().parse(text)
                  if field is not None)


def _catalog(code: str) -> dict:
    if not code or Path(code).name != code or code in (".", ".."):
        return {}
    try:
        data = json.loads((LANG_DIR / f"{code}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def languages() -> dict[str, str]:
    """Always offer English; add usable catalogs found on disk."""
    out = {DEFAULT: "English"}
    for path in sorted(LANG_DIR.glob("*.json")):
        if path.stem == DEFAULT:
            continue
        name = _catalog(path.stem).get("_name")
        if isinstance(name, str) and name.strip():
            out[path.stem] = name
    return out


def saved_language() -> str:
    try:
        data = json.loads(SETTINGS.read_text(encoding="utf-8"))
        code = data.get("language", DEFAULT) if isinstance(data, dict) else DEFAULT
        return code if isinstance(code, str) else DEFAULT
    except (OSError, ValueError):
        return DEFAULT


def use(code: str | None = None) -> None:
    """Load the selected catalog; use English if it is unavailable."""
    global _code
    code = code or saved_language()
    _code = code if code in languages() else DEFAULT
    _table.clear()
    if _code == DEFAULT:
        return
    for source, target in _catalog(_code).items():
        if source.startswith("_") or not isinstance(target, str):
            continue
        try:
            if _fields(source) == _fields(target):
                _table[source] = target
        except ValueError:
            continue


def tr(text, *args, **kwargs):
    """Translate a complete English message, then format its data values."""
    if _code is None:
        use()
    if not isinstance(text, str):
        return text
    translated = _table.get(text, text)
    if not args and not kwargs:
        return translated
    try:
        return translated.format(*args, **kwargs)
    except (ValueError, KeyError, IndexError):
        return text.format(*args, **kwargs)


def _translated(options):
    if not isinstance(options, dict):
        return options
    return {key: tr(value) if key in ("text", "label") else value for key, value in options.items()}


def install() -> None:
    """Translate Tk presentation options while retaining data and selection IDs."""
    if _code is None:
        use()
    if getattr(tk, "_cnk_igb_i18n", False):
        return
    tk._cnk_igb_i18n = True
    options = tk.Misc._options

    def widget_options(self, cnf, kw=None):
        return options(self, _translated(cnf), _translated(kw))
    tk.Misc._options = widget_options
    format_options = ttk._format_optdict

    def ttk_options(opts, script=False, ignore=None):
        return format_options(_translated(opts), script, ignore)
    ttk._format_optdict = ttk_options
    title = tk.Wm.wm_title

    def wm_title(self, string=None):
        return title(self, tr(string))
    tk.Wm.wm_title = tk.Wm.title = wm_title
    show = messagebox._show

    def show_message(title=None, message=None, _icon=None, _type=None, **options):
        if "detail" in options:
            options["detail"] = tr(options["detail"])
        return show(tr(title), tr(message), _icon, _type, **options)
    messagebox._show = show_message
    dialog = commondialog.Dialog.show

    def show_dialog(self, **options):
        for opts in (self.options, options):
            if "title" in opts:
                opts["title"] = tr(opts["title"])
            if "filetypes" in opts:
                opts["filetypes"] = [(tr(label), pattern) for label, pattern in opts["filetypes"]]
        return dialog(self, **options)
    commondialog.Dialog.show = show_dialog


def selector(parent, settings: dict, save, restart=None) -> ttk.Combobox:
    """Persist the selected installed language before restarting the editor."""
    langs = languages()
    code = _code if _code in langs else DEFAULT
    var = tk.StringVar(value=langs[code])
    box = ttk.Combobox(parent, textvariable=var, values=list(langs.values()), state="readonly", width=10)
    box.language_var = var  # Keep the Tcl variable alive after this function returns.
    if len(langs) == 1:
        box.configure(state="disabled")

    def picked(_event=None):
        code = list(langs)[box.current()]
        if code == (_code or DEFAULT):
            return
        settings["language"] = code
        save(settings)
        if not (restart and restart()):
            messagebox.showinfo(tr("Language"), tr("The language changes the next time the editor opens."), parent=parent)
    box.bind("<<ComboboxSelected>>", picked)
    return box
