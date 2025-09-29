# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for Auto Yolo Labeler."""

from pathlib import Path

block_cipher = None

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_README = PROJECT_ROOT / "sampro" / "checkpoints" / "README.txt"

extra_datas = [
    (str(CHECKPOINTS_README), "sampro/checkpoints"),
]


a = Analysis(
    ['Run.py'],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=extra_datas,
    hiddenimports=[],
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
    name='AutoYoloLabeler',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AutoYoloLabeler',
)
