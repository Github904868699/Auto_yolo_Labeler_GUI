# SAM 模型权重管理

应用会在启动时加载保存在 `~/.auto_yolo_labeler/config.json` 中配置的 SAM 模型权重路径。首次运行或配置缺失时，程序会回退到内置的 `sampro/checkpoints/` 目录。

## 桌面版 (PyInstaller) 发布包结构

使用提供的 `auto_yolo_labeler.spec` 打包后，发布目录会包含如下结构：

```
AutoYoloLabeler/
├── AutoYoloLabeler.exe
└── sampro/
    └── checkpoints/
        └── README.txt
```

将官方或自训练得到的 `.pt/.pth` 权重文件复制到 `sampro/checkpoints/` 目录即可。更新或替换模型时，确保旧文件被覆盖，重新启动程序即可生效。

## 在应用内更新模型

通过菜单 **File → 选择 SAM 模型文件** 可以手动指定任意位置的权重文件。选择成功后，路径会写入本地配置 (`~/.auto_yolo_labeler/config.json`)，下次启动会自动加载该文件。

当配置的权重文件不存在时，程序会弹出提示并引导用户检查 `sampro/checkpoints/` 目录或重新选择模型文件。
