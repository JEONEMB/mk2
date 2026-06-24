import ctypes as ct
import unittest
from types import SimpleNamespace

import numpy as np

from camera.synexens_camera import NativeSynexensBackend, _SYDeviceInfo, _SYIntrinsics


class _FakeSdk:
    def __init__(self) -> None:
        self.pointcloud_flags: list[bool] = []
        self.intrinsics_flags: list[bool] = []

    def GetDepthPointCloud(self, _device, _width, _height, _depth, _points, undistort):
        self.pointcloud_flags.append(bool(undistort))
        return 0

    def GetIntric(self, _device, _resolution, undistort, target):
        self.intrinsics_flags.append(bool(undistort))
        native = ct.cast(target, ct.POINTER(_SYIntrinsics)).contents
        native.fx, native.fy, native.cx, native.cy = 200.0, 200.0, 160.0, 120.0
        native.width, native.height = 320, 240
        return 0


class SynexensGeometryTests(unittest.TestCase):
    def test_undistorted_cloud_and_intrinsics_use_matching_sdk_flag(self) -> None:
        backend = NativeSynexensBackend(SimpleNamespace())
        backend._device = _SYDeviceInfo(device_id=7)
        backend._resolution_enum = 1
        backend._sdk = _FakeSdk()

        backend._get_depth_pointcloud(np.ones((2, 3), dtype=np.uint16), undistort=True)
        backend._get_depth_pointcloud(np.ones((2, 3), dtype=np.uint16), undistort=False)
        intrinsics = backend._get_intrinsics_for_frame(320, 240)

        self.assertEqual(backend._sdk.pointcloud_flags, [True, False])
        self.assertEqual(backend._sdk.intrinsics_flags, [True])
        self.assertEqual((intrinsics.width, intrinsics.height), (320, 240))


if __name__ == "__main__":
    unittest.main()
