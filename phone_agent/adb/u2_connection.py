"""UIAutomator2 连接缓存。

该模块只在自定义 rule 使用元素级操作时才会被懒加载。
没有安装 uiautomator2 时，普通坐标点击、截图和模型流程不受影响。
"""

from __future__ import annotations

from typing import Any

_devices: dict[str, Any] = {}


def get_u2_device(device_id: str | None = None) -> Any:
    """获取并缓存 UIAutomator2 设备连接。"""
    try:
        import uiautomator2 as u2
    except ImportError as exc:
        raise RuntimeError(
            "元素级 rule 需要安装 uiautomator2：pip install uiautomator2"
        ) from exc

    key = device_id or "__default__"
    if key not in _devices:
        _devices[key] = u2.connect(device_id) if device_id else u2.connect()
    return _devices[key]
