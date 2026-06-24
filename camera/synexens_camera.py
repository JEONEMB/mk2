"""Native Synexens CS20 adapter.

The supplied Synexens 4.2.5 SDK is a C/C++ DLL distribution, not a Python
package.  This module is therefore the sole owner of its ``ctypes`` ABI calls.
Every other package receives only SDK-independent ``DepthFrame`` objects.
"""

from __future__ import annotations

import ctypes as ct
import os
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from camera.frame_types import CameraInfo, CameraIntrinsics, DepthFrame


class CameraError(RuntimeError):
    """Raised for connection, streaming, and native-SDK problems."""


# Values and layouts come from the supplied C# demo's SYData*Define.cs files.
_SUCCESS = 0
_NO_FRAME = 20
_STREAM_DEPTH_IR = 4
_FRAME_DEPTH = 2
_FRAME_IR = 3
_RESOLUTIONS = {"320x240": 1, "640x480": 2, "960x540": 3, "1920x1080": 4, "1600x1200": 5, "800x600": 6}


class _SYDeviceInfo(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("device_id", ct.c_uint32),
        ("device_type", ct.c_int32),
        ("usb_bus", ct.c_uint32),
        ("usb_ports", ct.c_uint32 * 7),
        ("usb_port_count", ct.c_uint32),
        ("usb_device_address", ct.c_uint32),
        ("ip_address", ct.c_uint32),
    ]


class _SYFrameInfo(ct.Structure):
    _pack_ = 1
    _fields_ = [("frame_type", ct.c_int32), ("height", ct.c_int32), ("width", ct.c_int32)]


class _SYFrameData(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("frame_count", ct.c_int32),
        ("frame_info", ct.c_void_p),
        ("data", ct.c_void_p),
        ("buffer_length", ct.c_int32),
    ]


class _SYIntrinsics(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("fov", ct.c_float * 2),
        ("coefficients", ct.c_float * 5),
        ("fx", ct.c_float),
        ("fy", ct.c_float),
        ("cx", ct.c_float),
        ("cy", ct.c_float),
        ("width", ct.c_int32),
        ("height", ct.c_int32),
        ("tof_to_rgb_rotation", ct.c_float * 9),
        ("tof_to_rgb_translation", ct.c_float * 3),
    ]


class _SYPointCloudData(ct.Structure):
    _pack_ = 1
    _fields_ = [("x", ct.c_float), ("y", ct.c_float), ("z", ct.c_float)]


class CameraBackend(Protocol):
    def open(self) -> CameraInfo: ...
    def start(self) -> None: ...
    def read_frame(self, timeout_ms: int) -> DepthFrame | None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class NativeSynexensBackend:
    """``ctypes`` binding for the C API used by the official C# demo.

    The device-owned frame buffer is copied before this method returns, making
    it safe for the SDK to publish the next frame immediately afterwards.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._sdk: ct.CDLL | None = None
        self._dll_directory_handles: list[Any] = []
        self._device: _SYDeviceInfo | None = None
        self._intrinsics: CameraIntrinsics | None = None
        self._resolution_enum: int | None = None
        self._initialized = False
        self._streaming = False
        self._has_ir = False

    def open(self) -> CameraInfo:
        root = self._resolve_sdk_root()
        dll_path = root / "SynexensSDK.dll"
        self._add_dll_directories(root)
        try:
            self._sdk = ct.CDLL(str(dll_path))  # C# demo declares CallingConvention.Cdecl.
        except OSError as exc:
            raise CameraError(f"could not load {dll_path}: {exc}") from exc
        self._bind_api()
        self._check(self._sdk.InitSDK(), "InitSDK")
        self._initialized = True

        count = ct.c_int(0)
        self._check(self._sdk.FindDevice(ct.byref(count), None), "FindDevice(count)")
        index = int(self.config.device_index)
        if count.value <= index:
            raise CameraError(f"CS20 device index {index} was not found; detected {count.value} device(s)")
        devices = (_SYDeviceInfo * count.value)()
        self._check(self._sdk.FindDevice(ct.byref(count), devices), "FindDevice(list)")
        self._device = devices[index]
        self._check(self._sdk.OpenDevice(ct.byref(self._device)), "OpenDevice")
        serial = self._serial_number()
        return CameraInfo(
            device_id=str(self._device.device_id),
            model=f"Synexens device type {self._device.device_type}",
            serial_number=serial,
            sdk_version=self._sdk_version(),
        )

    def start(self) -> None:
        device_id = self._device_id()
        resolution = self._resolution_value(self.config.default_resolution)
        # The C# demo sets only DEPTH resolution; DEPTHIR shares that resolution.
        self._check(self._sdk.SetFrameResolution(device_id, _FRAME_DEPTH, resolution), "SetFrameResolution(depth)")
        # CS20-P is depth-only; other CS20 variants support the documented DEPTHIR stream.
        self._has_ir = bool(self.config.enable_ir and self._device is not None and self._device.device_type != 5)
        stream_type = _STREAM_DEPTH_IR if self._has_ir else 2  # DEPTH
        self._check(self._sdk.StartStreaming(device_id, stream_type), "StartStreaming")
        self._streaming = True
        self._resolution_enum = resolution
        native = _SYIntrinsics()
        self._check(self._sdk.GetIntric(device_id, resolution, False, ct.byref(native)), "GetIntric(depth)")
        self._intrinsics = CameraIntrinsics(
            float(native.fx), float(native.fy), float(native.cx), float(native.cy), int(native.width), int(native.height)
        )

    def set_resolution(self, resolution_name: str) -> None:
        """Change stream resolution and refresh its matching intrinsics."""

        device_id = self._device_id()
        resolution = self._resolution_value(resolution_name)
        was_streaming = self._streaming
        if was_streaming:
            self._check(self._sdk.StopStreaming(device_id), "StopStreaming(resolution change)")
            self._streaming = False
        self._check(self._sdk.SetFrameResolution(device_id, _FRAME_DEPTH, resolution), "SetFrameResolution(depth)")
        if was_streaming:
            stream_type = _STREAM_DEPTH_IR if self._has_ir else 2  # DEPTH
            self._check(self._sdk.StartStreaming(device_id, stream_type), "StartStreaming(resolution change)")
            self._streaming = True
        self._resolution_enum = resolution
        native = _SYIntrinsics()
        self._check(self._sdk.GetIntric(device_id, resolution, False, ct.byref(native)), "GetIntric(depth)")
        self._intrinsics = CameraIntrinsics(
            float(native.fx), float(native.fy), float(native.cx), float(native.cy), int(native.width), int(native.height)
        )

    def read_frame(self, timeout_ms: int) -> DepthFrame | None:
        del timeout_ms  # GetLastFrameData is non-blocking in the vendor API.
        if not self._streaming or self._intrinsics is None:
            raise CameraError("camera stream has not been started")
        frame_ptr = ct.c_void_p()
        code = int(self._sdk.GetLastFrameData(self._device_id(), ct.byref(frame_ptr)))
        if code == _NO_FRAME or not frame_ptr.value:
            return None
        self._check(code, "GetLastFrameData")
        frame = ct.cast(frame_ptr, ct.POINTER(_SYFrameData)).contents
        if frame.frame_count <= 0 or not frame.frame_info or not frame.data:
            return None
        raw_depth, ir = self._copy_depth_and_ir(frame)
        if raw_depth is None:
            return None
        pointcloud = self._get_depth_pointcloud(raw_depth)
        # The native Z coordinates are metric SDK output. They replace raw
        # uint16 depth, which is not guaranteed to be millimetres.
        depth = pointcloud[..., 2].astype(np.float32)
        depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
        if depth.shape != (self._intrinsics.height, self._intrinsics.width):
            # Resolution can change through device settings; fetch current calibration.
            self._intrinsics = self._get_intrinsics_for_frame(depth.shape[1], depth.shape[0])
        return DepthFrame(
            depth,
            self._intrinsics,
            ir,
            pointcloud.astype(np.float32),
            time.time(),
            f"{depth.shape[1]}x{depth.shape[0]}",
        )

    def stop(self) -> None:
        if self._sdk is not None and self._device is not None and self._streaming:
            self._sdk.StopStreaming(self._device.device_id)
        self._streaming = False

    def close(self) -> None:
        self.stop()
        if self._sdk is not None and self._device is not None:
            self._sdk.CloseDevice(self._device.device_id)
        if self._sdk is not None and self._initialized:
            self._sdk.UnInitSDK()
        self._initialized = False
        self._device = None
        self._intrinsics = None
        self._sdk = None
        self._dll_directory_handles.clear()

    def _copy_depth_and_ir(self, frame: _SYFrameData) -> tuple[np.ndarray | None, np.ndarray | None]:
        infos = ct.cast(frame.frame_info, ct.POINTER(_SYFrameInfo))
        offset = 0
        depth: np.ndarray | None = None
        ir: np.ndarray | None = None
        for index in range(frame.frame_count):
            info = infos[index]
            pixel_count = info.width * info.height
            if pixel_count <= 0:
                return None, None
            # DEPTHIR frames are 16-bit. The explicit RGB case keeps offsets correct
            # should a device append an RGB frame in a later SDK configuration.
            bytes_per_pixel = 3 if info.frame_type == 4 else 2
            byte_count = pixel_count * bytes_per_pixel
            if offset + byte_count > frame.buffer_length:
                raise CameraError("SDK returned a truncated frame buffer")
            if info.frame_type in (_FRAME_DEPTH, _FRAME_IR):
                raw = ct.string_at(int(frame.data) + offset, pixel_count * 2)
                image = np.frombuffer(raw, dtype="<u2").copy().reshape(info.height, info.width)
                if info.frame_type == _FRAME_DEPTH:
                    depth = image
                else:
                    ir = image
            offset += byte_count
        return depth, ir

    def _get_depth_pointcloud(self, raw_depth: np.ndarray) -> np.ndarray:
        depth = np.ascontiguousarray(raw_depth, dtype=np.uint16)
        height, width = depth.shape
        points = np.empty((height, width, 3), dtype=np.float32)
        self._check(
            self._sdk.GetDepthPointCloud(
                self._device_id(),
                width,
                height,
                ct.c_void_p(depth.ctypes.data),
                ct.c_void_p(points.ctypes.data),
                False,
            ),
            "GetDepthPointCloud",
        )
        return points

    def _get_intrinsics_for_frame(self, width: int, height: int) -> CameraIntrinsics:
        resolution = self._resolution_enum or self._resolution_value(f"{width}x{height}")
        native = _SYIntrinsics()
        self._check(self._sdk.GetIntric(self._device_id(), resolution, False, ct.byref(native)), "GetIntric(depth)")
        intrinsics = CameraIntrinsics(
            float(native.fx), float(native.fy), float(native.cx), float(native.cy), int(native.width), int(native.height)
        )
        if intrinsics.width != width or intrinsics.height != height:
            raise CameraError(
                f"SDK intrinsics are {intrinsics.width}x{intrinsics.height}, but frame is {width}x{height}"
            )
        return intrinsics

    def _bind_api(self) -> None:
        assert self._sdk is not None
        self._sdk.InitSDK.argtypes, self._sdk.InitSDK.restype = [], ct.c_int
        self._sdk.UnInitSDK.argtypes, self._sdk.UnInitSDK.restype = [], ct.c_int
        self._sdk.FindDevice.argtypes, self._sdk.FindDevice.restype = [ct.POINTER(ct.c_int), ct.POINTER(_SYDeviceInfo)], ct.c_int
        self._sdk.OpenDevice.argtypes, self._sdk.OpenDevice.restype = [ct.POINTER(_SYDeviceInfo)], ct.c_int
        self._sdk.CloseDevice.argtypes, self._sdk.CloseDevice.restype = [ct.c_uint32], ct.c_int
        self._sdk.StartStreaming.argtypes, self._sdk.StartStreaming.restype = [ct.c_uint32, ct.c_int], ct.c_int
        self._sdk.StopStreaming.argtypes, self._sdk.StopStreaming.restype = [ct.c_uint32], ct.c_int
        self._sdk.SetFrameResolution.argtypes, self._sdk.SetFrameResolution.restype = [ct.c_uint32, ct.c_int, ct.c_int], ct.c_int
        self._sdk.GetLastFrameData.argtypes, self._sdk.GetLastFrameData.restype = [ct.c_uint32, ct.POINTER(ct.c_void_p)], ct.c_int
        self._sdk.GetIntric.argtypes, self._sdk.GetIntric.restype = [ct.c_uint32, ct.c_int, ct.c_bool, ct.POINTER(_SYIntrinsics)], ct.c_int
        self._sdk.GetDepthPointCloud.argtypes, self._sdk.GetDepthPointCloud.restype = [
            ct.c_uint32,
            ct.c_int,
            ct.c_int,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_bool,
        ], ct.c_int
        self._sdk.GetDeviceSN.argtypes, self._sdk.GetDeviceSN.restype = [ct.c_uint32, ct.POINTER(ct.c_int), ct.c_void_p], ct.c_int
        self._sdk.GetSDKVersion.argtypes, self._sdk.GetSDKVersion.restype = [ct.POINTER(ct.c_int), ct.c_void_p], ct.c_int

    def _serial_number(self) -> str | None:
        length = ct.c_int(0)
        if int(self._sdk.GetDeviceSN(self._device_id(), ct.byref(length), None)) != _SUCCESS or length.value <= 0:
            return None
        buffer = ct.create_string_buffer(length.value + 1)
        if int(self._sdk.GetDeviceSN(self._device_id(), ct.byref(length), buffer)) != _SUCCESS:
            return None
        return buffer.value.decode(errors="replace")

    def _sdk_version(self) -> str | None:
        length = ct.c_int(0)
        if int(self._sdk.GetSDKVersion(ct.byref(length), None)) != _SUCCESS or length.value <= 0:
            return None
        buffer = ct.create_string_buffer(length.value + 1)
        if int(self._sdk.GetSDKVersion(ct.byref(length), buffer)) != _SUCCESS:
            return None
        return buffer.value.decode(errors="replace")

    def _device_id(self) -> int:
        if self._device is None:
            raise CameraError("camera has not been opened")
        return int(self._device.device_id)

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if int(code) != _SUCCESS:
            raise CameraError(f"{operation} failed with Synexens error code {int(code)}")

    @staticmethod
    def _resolution_value(name: str) -> int:
        try:
            return _RESOLUTIONS[name]
        except KeyError as exc:
            raise CameraError(f"unsupported Synexens resolution: {name}") from exc

    def _add_dll_directories(self, root: Path) -> None:
        if not hasattr(os, "add_dll_directory"):
            return
        for path in (root, root / "bin", root / "dll"):
            if path.is_dir():
                self._dll_directory_handles.append(os.add_dll_directory(str(path)))

    def _resolve_sdk_root(self) -> Path:
        roots: list[Path] = []
        configured = getattr(self.config, "sdk_root", None)
        if configured is not None:
            roots.append(Path(configured))
        roots.extend(Path(path) for path in getattr(self.config, "fallback_sdk_roots", ()))
        for root in roots:
            if (root / "SynexensSDK.dll").is_file():
                return root
        checked = ", ".join(str(root / "SynexensSDK.dll") for root in roots)
        raise CameraError(f"SynexensSDK.dll was not found. Checked: {checked}")


class SyntheticCameraBackend:
    """Deterministic CS20-like source for pipeline testing without hardware."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width, self.height = width, height
        self._running = False
        self._rng = np.random.default_rng(20260623)
        self.intrinsics = CameraIntrinsics(580.0, 580.0, width / 2, height / 2, width, height)

    def open(self) -> CameraInfo:
        return CameraInfo("demo-0", "Synthetic CS20", "DEMO", "demo")

    def start(self) -> None:
        self._running = True

    def set_resolution(self, resolution_name: str) -> None:
        try:
            width, height = (int(value) for value in resolution_name.split("x", maxsplit=1))
        except ValueError as exc:
            raise CameraError(f"unsupported synthetic resolution: {resolution_name}") from exc
        if (width, height) not in ((320, 240), (640, 480)):
            raise CameraError(f"unsupported synthetic resolution: {resolution_name}")
        self.width, self.height = width, height
        # Keep the demo camera's field of view constant as resolution changes.
        focal = 580.0 * width / 640.0
        self.intrinsics = CameraIntrinsics(focal, focal, width / 2, height / 2, width, height)

    def read_frame(self, timeout_ms: int) -> DepthFrame:
        if not self._running:
            raise CameraError("synthetic stream has not been started")
        yy, xx = np.mgrid[: self.height, : self.width]
        depth = 720.0 + 0.012 * (xx - self.width / 2) - 0.006 * (yy - self.height / 2)
        box = (
            (xx >= self.width * 0.344)
            & (xx < self.width * 0.656)
            & (yy >= self.height * 0.344)
            & (yy < self.height * 0.656)
        )
        depth[box] -= 120.0
        depth += self._rng.normal(0.0, 1.0, depth.shape)
        ir = self._rng.integers(25, 140, size=depth.shape, dtype=np.uint8)
        return DepthFrame(
            depth_mm=depth.astype(np.float32),
            intrinsics=self.intrinsics,
            ir_image=ir,
            timestamp=time.time(),
            resolution_name=f"{self.width}x{self.height}",
        )

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._running = False


class SynexensCamera:
    """Public camera facade; lower layers never see the native SDK."""

    def __init__(self, sdk_config: Any, depth_config: Any | None = None, backend: CameraBackend | None = None) -> None:
        self.sdk_config = sdk_config
        self.depth_config = depth_config
        self._backend: CameraBackend = backend or NativeSynexensBackend(sdk_config)
        self.info: CameraInfo | None = None
        self._opened = False
        self._started = False

    def open(self) -> CameraInfo:
        self.info = self._backend.open()
        self._opened = True
        return self.info

    def start(self) -> None:
        if not self._opened:
            raise CameraError("call open() before start()")
        self._backend.start()
        self._started = True

    def read(self) -> DepthFrame | None:
        if not self._started:
            raise CameraError("call start() before read()")
        return self._backend.read_frame(self.sdk_config.frame_timeout_ms)

    def set_resolution(self, resolution_name: str) -> None:
        target = getattr(self._backend, "set_resolution", None)
        if not callable(target):
            raise CameraError("the active camera backend does not support changing resolution")
        target(resolution_name)

    def collect_frames(self, count: int) -> list[DepthFrame]:
        frames: list[DepthFrame] = []
        while len(frames) < count:
            frame = self.read()
            if frame is not None:
                frames.append(frame)
        return frames

    def stop(self) -> None:
        if self._started:
            self._backend.stop()
            self._started = False

    def close(self) -> None:
        self._backend.close()
        self._opened = False
