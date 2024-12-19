# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['squat_counter.py'],
    pathex=[],
    binaries=[],
    datas=[('ding.wav', '.'), ('C:\\Users\\kalinga\\Downloads\\Mine\\All Codes\\squat-counter\\venv\\lib\\site-packages\\mediapipe', 'mediapipe/')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='squat_counter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
