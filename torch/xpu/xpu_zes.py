"""
Level Zero (ZES) helpers for XPU device enumeration.

These functions mirror the role of the NVML helpers in torch/cuda/__init__.py:
they enumerate Intel GPU devices via the Level Zero loader library without
initialising the full XPU/SYCL runtime, so they are safe to call before fork.
"""

from __future__ import annotations

import os
import warnings
from typing import cast


_HAS_PYZES = False
_PYZES_ERR = None
try:
    import pyzes  # type: ignore[import]

    _HAS_PYZES = True
except ModuleNotFoundError:
    pass
except ImportError as err:
    _PYZES_ERR = err  # installed but import fails for another reason


def _load_ze_loader():
    """Return a ctypes handle to libze_loader, or raise OSError if unavailable."""
    from ctypes import CDLL

    try:
        return CDLL("libze_loader.so.1")
    except OSError:
        return CDLL("libze_loader.so")


def _raw_device_count_zes() -> int:
    r"""Return number of XPU devices as reported by Level Zero.

    Returns -1 if Level Zero discovery/initialization failed.
    """
    from ctypes import byref, c_uint32, c_void_p

    try:
        ze_h = _load_ze_loader()
    except OSError:
        warnings.warn("Can't load libze_loader", stacklevel=2)
        return -1

    ZE_INIT_FLAG_GPU_ONLY = 1
    rc = ze_h.zesInit(0)
    if rc != 0:
        warnings.warn("Can't initialize Level Zero", stacklevel=2)
        return -1

    driver_count = c_uint32(0)
    rc = ze_h.zesDriverGet(byref(driver_count), None)
    print(f"Level Zero driver count: {driver_count.value}")
    if rc != 0 or driver_count.value == 0:
        warnings.warn("Can't get Level Zero driver count", stacklevel=2)
        return -1

    DriverArray = c_void_p * driver_count.value
    drivers = DriverArray()
    rc = ze_h.zesDriverGet(byref(driver_count), drivers)
    if rc != 0:
        warnings.warn("Can't get Level Zero driver handles", stacklevel=2)
        return -1

    total = 0
    for i in range(driver_count.value):
        dev_count = c_uint32(0)
        rc = ze_h.zesDeviceGet(drivers[i], byref(dev_count), None)
        print(f"Level Zero driver {i} device count: {dev_count.value}")
        if rc == 0:
            total += dev_count.value
    del ze_h
    return total


def _raw_device_uuid_zes() -> list[str] | None:
    r"""Return list of XPU device UUIDs as reported by Level Zero.

    Returns None if Level Zero discovery/initialization failed.
    """
    import struct
    from ctypes import byref, c_uint32, c_void_p, create_string_buffer

    try:
        ze_h = _load_ze_loader()
    except OSError:
        warnings.warn("Can't load libze_loader", stacklevel=2)
        return None

    ZE_INIT_FLAG_GPU_ONLY = 1
    rc = ze_h.zeInit(ZE_INIT_FLAG_GPU_ONLY)
    if rc != 0:
        warnings.warn("Can't initialize Level Zero", stacklevel=2)
        return None

    driver_count = c_uint32(0)
    rc = ze_h.zeDriverGet(byref(driver_count), None)
    if rc != 0 or driver_count.value == 0:
        warnings.warn("Can't get Level Zero driver count", stacklevel=2)
        return None

    DriverArray = c_void_p * driver_count.value
    drivers = DriverArray()
    rc = ze_h.zeDriverGet(byref(driver_count), drivers)
    if rc != 0:
        warnings.warn("Can't get Level Zero driver handles", stacklevel=2)
        return None

    uuids: list[str] = []
    for i in range(driver_count.value):
        dev_count = c_uint32(0)
        rc = ze_h.zeDeviceGet(drivers[i], byref(dev_count), None)
        if rc != 0:
            warnings.warn("Can't get Level Zero device count", stacklevel=2)
            return None
        DevArray = c_void_p * dev_count.value
        devs = DevArray()
        rc = ze_h.zeDeviceGet(drivers[i], byref(dev_count), devs)
        if rc != 0:
            warnings.warn("Can't get Level Zero device handles", stacklevel=2)
            return None
        # ze_device_properties_t layout (64-bit):
        #   stype  (uint32, offset 0)
        #   _pad   (uint32, offset 4)
        #   pNext  (void*,  offset 8)
        #   uuid   (16 bytes, offset 16)
        for j in range(dev_count.value):
            buf = create_string_buffer(512)
            # ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x1
            struct.pack_into("I", buf, 0, 0x1)
            rc = ze_h.zeDeviceGetProperties(devs[j], buf)
            if rc != 0:
                warnings.warn("Can't get Level Zero device properties", stacklevel=2)
                return None
            uuid_bytes = buf.raw[16:32]
            uuids.append(uuid_bytes.hex())
    del ze_h
    return uuids


def _parse_ze_affinity_mask() -> list[int] | list[str]:
    r"""Parse ZE_AFFINITY_MASK environment variable.

    Supports ordinal lists (``"0,1"``) and tile-qualified ordinals
    (``"0.0,1.0"`` → ``[0, 1]``).  Returns a list of integer device
    ordinals, or up to 64 ordinals if the variable is unset.
    """
    var = os.getenv("ZE_AFFINITY_MASK")
    if var is None:
        return list(range(64))

    rc: list[int] = []
    for elem in var.split(","):
        # Strip tile suffix: "0.0" → "0"
        device_part = elem.strip().split(".")[0]
        try:
            x = int(device_part)
        except ValueError:
            break
        if x < 0:
            break
        if x not in rc:
            rc.append(x)
    return rc


def _device_count_zes() -> int:
    r"""Return number of XPU devices taking ZE_AFFINITY_MASK into account.

    Returns a negative value if Level Zero discovery or initialization failed.
    Mirrors ``_device_count_nvml()`` in ``torch/cuda/__init__.py``.
    """
    visible_devices = _parse_ze_affinity_mask()
    print(f"ZE_AFFINITY_MASK visible devices: {visible_devices}")
    if not visible_devices:
        return 0
    raw_cnt = _raw_device_count_zes()
    print(f"Level Zero raw device count: {raw_cnt}")
    if raw_cnt <= 0:
        return raw_cnt
    # Trim visible list to actually available devices
    for idx, val in enumerate(cast(list[int], visible_devices)):
        if cast(int, val) >= raw_cnt:
            return idx
    return len(visible_devices)


print(_device_count_zes())