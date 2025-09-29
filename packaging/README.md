# Packaging Auto Yolo Labeler

This directory stores helper assets for building a standalone Windows
executable with [PyInstaller](https://pyinstaller.org/).

## Prerequisites

1. Install the Python dependencies listed in `requirements.txt`. The list
   already contains the runtime stack required by the application when frozen
   (PyQt5, OpenCV, Torch, etc.) as well as `pyinstaller>=6.10.0` for the
   packaging step.
2. Download the Segment Anything v2 (SAM2) model weights and place them in
   `sampro/checkpoints/`. The application expects the file
   `sam2.1_hiera_large.pt` by default; override the `SAM2_CHECKPOINT`
   environment variable if you use another filename.

## Build

From the project root:

```bash
pyinstaller packaging/auto_yolo_labeler.spec
```

The spec file bundles the GUI icons together with the `segment`, `util`, and
`sampro` packages so that model weights and helper modules remain available at
runtime. The resulting executable is written to `dist/AutoYoloLabeler.exe`.

## Testing

After building, copy the `dist` folder to a clean Windows machine and launch
`AutoYoloLabeler.exe` to verify that the GUI opens and the automatic labelling
pipeline runs end-to-end with your SAM model checkpoint.
