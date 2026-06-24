import unittest

import numpy as np

from calibration.platform_calibrator import PlatformCalibrator
from config import CalibrationConfig


class PlatformProposalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CalibrationConfig()
        self.calibrator = PlatformCalibrator(self.config)
        self.height, self.width = 240, 320
        yy, xx = np.mgrid[: self.height, : self.width]
        self.plane = np.dstack(
            ((xx - self.width / 2) * 2.0, (yy - self.height / 2) * 2.0, np.full((self.height, self.width), 700.0))
        )

    def test_single_plane_produces_guarded_verified_roi(self) -> None:
        proposal = self.calibrator.propose_platform([self.plane] * 3)

        self.assertTrue(proposal.accepted)
        self.assertEqual(proposal.outer_roi, (0, 0, self.width, self.height))
        self.assertEqual(proposal.measurement_roi, (16, 16, self.width - 32, self.height - 32))
        self.assertEqual(proposal.roi_coverage, 1.0)

    def test_foreground_region_is_not_included_in_suggested_rectangle(self) -> None:
        grid = self.plane.copy()
        grid[20:80, 120:200, 2] = 600.0

        proposal = self.calibrator.propose_platform([grid] * 3)

        self.assertTrue(proposal.accepted, proposal.failure_reason)
        assert proposal.outer_roi is not None
        x, y, width, height = proposal.outer_roi
        overlaps_foreground = x < 200 and x + width > 120 and y < 80 and y + height > 20
        self.assertFalse(overlaps_foreground)

    def test_nearby_islands_remain_separate_raw_components(self) -> None:
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[20:180, 20:120] = True
        mask[20:180, 140:240] = True  # 20 px gap: formerly bridged by 41 px closing.

        components = self.calibrator._raw_plane_components(mask)

        self.assertEqual(len(components), 2)


if __name__ == "__main__":
    unittest.main()
