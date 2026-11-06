# CNK IGB Editor

Editor and viewer for **Crash Nitro Kart PS2, Xbox retail and Xbox prototype**
Alchemy files. Requires Python 3.10+ with Tkinter. The 3D view uses Windows
OpenGL; no additional Python packages are required.

## Starting the editor

Double-click `CNK IGB Editor.pyw`, or run from the repository root:

```powershell
python tools/cnk_igb_editor/cnk_igb_gui.py
python tools/cnk_igb_editor/cnk_igb_gui.py model.igb
```

Open **ASSETS.GFC** or **ASSETS.GOB**, keeping both in the same directory.
The archive identifies the platform without converting its files.
The browser lists every file and searches its full path. **Characters**,
**Tracks**, **Images** and **Animations** include karts, menu models, props and
effects. **Reset** clears the search; **All** restores every category.
PNG and TGA files open as images. Other files have text or hexadecimal previews
and can be extracted. The selector above the sidebar provides access to all tabs.

## Languages

All source code, comments, messages and documentation use English. English is
built in and requires no language file. Optional catalogs live in `lang/`;
for example `lang/it.json` contains the English-to-Italian translations.

Use the language selector to choose English or an installed catalog. The choice
is saved and the editor restarts in that language. If `lang/` is empty, missing,
or contains no valid catalogs, only English is offered. A saved language whose
catalog has been removed also falls back to English. Missing translations and
invalid format placeholders fall back to the original English message.

A catalog is a UTF-8 JSON object containing `_name` for its display name and
English messages as keys. Messages containing data use numbered placeholders
such as `{0}` or `{1:.2f}`; translations must retain their fields and formats.
Values such as filenames, mesh names and bone names are inserted after
translation and are never translated themselves. English keys are stable message
identifiers, not fragments for search-and-replace. No English catalog is needed.

## Content and controls

- **Meshes:** all parts, LODs, rigid attachments and shadow volumes. The best LOD
  is selected initially; other meshes remain available.
- **Bones:** hierarchy, rest positions, matrix indices and skin weights.
- **Animations:** play, pause, speed and timeline. Rotation, translation and scale
  channels include facial bones. Transform sequences also animate rigid objects.
  Camera/helper-only files show paths, axes and the current animated position.
- **Face / pose:** actual facial bones (jaw, eyes, eyebrows, ears and others).
  Rotations and translations overlay the current pose and propagate to children.
  Disable the facial filter to control any bone. These are **preview edits**;
  they are not saved to IGB files or animations.
- **Blendshapes:** morph targets imported from glTF. Original CNK character
  expressions use facial bones rather than morph targets.
- **Textures:** preview, export PNG and replace images. Supports PS2 palettes,
  Xbox DXT1/DXT3/DXT5 and alpha. DXT recompression is lossy; unchanged images
  retain their original bytes. **Enlarge…** opens a scrollable 100–800% atlas.
  **3D filtering** switches between linear and pixel sampling. These controls
  affect the preview only. Font-only files such as `bane.igb` and `fonts.igb`
  automatically show their atlas on a dark background. Glyph metrics remain
  available in **IGB data**; text layout is not simulated.
- **IGB data:** every serialized class, object and memory block, including
  materials, lights, metadata, events and effect structures. Search by class,
  index or string; double-click references to follow them. Memory blocks can be
  paged and exported in full as `.bin`.

Left-drag to orbit, right/middle-drag to pan, wheel to zoom, double-click or **F**
to frame the model. Disable weight coloring to inspect textures while selecting
bones.

## Editing and exporting

Export glTF/GLB for Blender, preserve mesh and bone names, then import the edited
glTF over the original IGB and choose **Save .igb…**. Meshes, weights and textures
are replaced; the original skeleton and animations are preserved. Xbox indices
are retained when topology is unchanged and rebuilt when it changes. OBJ exports
geometry only. Exports include visible meshes.

Animation exports to glTF cover the skeleton. Camera and rigid scene transform
sequences currently play only in the viewer.

## Xbox prototype builds

Prototype IGBs can assign different slot numbers to skeletons, bones, bindings,
tracks and skins. The reader uses the field order and types declared in each
file, so conversion to a retail build is unnecessary. External directory
references, including the one in `ui/trophy/citadel.igb`, are visible in **IGB
data**, preserved when saving and do not consume embedded object payloads.
Resources referenced externally are not generated by the viewer.

Prototype Xbox geometry may omit separate index arrays. When an archive is
opened, its platform also determines the vertex color scale for these meshes.
Imports preserve native triangle lists and original weights when equivalent to
the glTF weights. PNG and indexed TGA images, including RLE, are supported.
Boss L8 images and flag RGBA5551 textures are decoded. Unknown image formats
produce an explicit error instead of a silent gray placeholder. Full-range
palette alpha retains partial transparency separately from PS2 alpha in 0–128.
Prototype podium Cortex contains a **256×256** texture; the inspected PS2/Xbox
retail versions contain a **128×128** texture.

## Preview fidelity and limits

Images are displayed at their original resolution, without additional reduction.
Zoom and filtering cannot add missing source detail. Atlas colors and seams may
remain visible with lighting disabled.

The viewer uses a camera-relative preview light. Original scene lights,
materials and game-specific effects are not reproduced in full. Disable
**Lighting** to inspect texture colors. Matching geometry and UVs does not
establish frame-by-frame visual equivalence with the original game.

The inspector also exposes data requiring the game engine: scripts, particle
behaviors, custom shaders and runtime effects are not simulated. Some files ship
NaN vertex placeholders for generated effects; undefined triangles are omitted
from the preview while original bytes remain available in the inspector.
IGB versions 4 and 5 found in the supplied archives are supported. The viewer
opens extracted archives and loose files. `cnk_iso.py` handles PS2 ISO9660 disc
images, not Xbox XDVDFS images.

## Verification

```powershell
python -m unittest discover -s tests -p test_igb.py -v
python -m unittest discover -s tests -p test_cnk_i18n.py -v
python tools/cnk_igb_editor/verify_archive.py path/ASSETS.GFC --contents --report report.json
python tools/cnk_igb_editor/verify_viewer.py path/ASSETS.GFC --report viewer.json
```

Game-data tests use `CNK_ASSETS` (PS2), `CNK_XBOX_ASSETS` (Xbox retail) and
`CNK_PROTO_ASSETS` (Xbox prototype), or remembered archive paths where applicable.
Integration tests are skipped when their fixtures are unavailable.
`verify_archive.py` checks every IGB, byte round trips, indices, textures and
three sampled poses per animation. It compares loaded bones, animations and
tracks with their declarations to detect silently skipped content.
`--characters` limits checks to characters, karts and menu models. `--contents`
also checks block checksums, reads every archive entry and decodes PNG/TGA files.

`verify_viewer.py` checks Tk callbacks and OpenGL errors on Windows, sampling
three frames per animation. `--all` includes scenery and effects; `--previews
DIR` saves selected previews. Both scripts accept `--filter substring`.
`verify_viewer.py --samples 30` samples more frames per clip and also verifies
image-only files. Neither tool changes game files or editor settings. Checks
establish structural validity and stability, not complete visual fidelity.

DXT layout: [Microsoft BC1/BC2/BC3 documentation](https://learn.microsoft.com/en-us/windows/uwp/graphics-concepts/block-compression).


### AI-Assisted Development

Parts of this project may have been developed with the assistance of AI tools. All AI-assisted contributions are subject to human review and validation.

AI-generated content may contain errors or inaccuracies. The project maintainers retain responsibility for the final implementation and maintenance.
