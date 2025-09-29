# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

block_cipher = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIRS = [
    "GUI/icons",
    "segment",
    "util",
    "sampro",
]


def _collect_resource_tree():
    datas = []
    for relative in RESOURCE_DIRS:
        source_root = PROJECT_ROOT / relative
        if not source_root.exists():
            continue
        for path in source_root.rglob("*"):
            if path.is_file():
                target = Path(relative) / path.relative_to(source_root)
                datas.append((str(path), str(target)))
    return datas


def _collect_hiddenimports():
    try:
        from PyInstaller.utils.hooks import collect_submodules  # type: ignore
    except Exception:  # pragma: no cover - PyInstaller is only available during packaging
        return []

    hidden = []
    for package in ("sampro", "segment"):
        try:
            hidden.extend(collect_submodules(package))
        except Exception:
            continue
    # Preserve ordering but drop duplicates
    seen = set()
    result = []
    for name in hidden:
        if name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


datas = _collect_resource_tree()
hiddenimports = _collect_hiddenimports()

pathex = [str(PROJECT_ROOT)]


a = Analysis(
    [str(PROJECT_ROOT / "Run.py")],
    pathex=pathex,
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="AutoYoloLabeler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "GUI/icons/logo.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AutoYoloLabeler",
)
