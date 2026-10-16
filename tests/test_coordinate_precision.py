"""Regression check for wide AMP cost-volume lookup rows."""
import types
import unittest

import torch

from s2m2.core.model.s2m2 import S2M2


class CoordinatePrecisionTest(unittest.TestCase):
    def test_wide_half_features_preserve_cost_volume_rows(self):
        # Exercise coordinate construction in the real S2M2.forward and both real
        # CostVolume samplers. Learned heads are stubs so the answer is exact.
        width = 2056
        features = torch.zeros(2, 1, 1, width, dtype=torch.float16)
        x = torch.arange(width, dtype=torch.float32)
        disparity = (x - 16).reshape(1, 1, 1, width)
        signs = (2 * (torch.arange(width) % 2) - 1).float()
        volume = signs.reshape(1, 1, width, 1).expand(1, 1, width, 32).contiguous()
        observed = []

        def refine(hidden, ctx, disp, conf, occ, cv):
            observed.extend(cv(disp))
            return hidden, disp, conf, occ

        stub = types.SimpleNamespace(
            normalize_img=lambda a, b: (a, b),
            extract_feature=lambda a, b: (features, features),
            feat_pyramid=lambda f: (f, f, f, f),
            transformer=lambda *f: features,
            disp_init=lambda f: (disparity, torch.ones_like(disparity),
                                 torch.ones_like(disparity), volume),
            global_refiner=lambda f, d, c: d,
            feat_fusion_layer=lambda a, b: a,
            ctx_feat=lambda f: f,
            refiner=refine,
            use_positivity=False,
            refine_iter=1,
            upsample_mask_4x_refine=lambda *args: None,
            upsample_mask_1x=lambda *args: None,
            upsample4x=lambda x, weights: x,
            upsample1x=lambda x, weights: x,
            output_upsample=False,
        )
        S2M2.forward(stub, torch.empty(0), torch.empty(0))
        self.assertEqual(len(observed), 2)
        for correlations in observed:
            expected = signs.reshape(1, 1, 1, width).expand_as(correlations)
            # FP32 grid normalization introduces sub-0.001 interpolation error.
            # Casting x=2049 to half instead selects the opposite sign (error 2).
            torch.testing.assert_close(correlations, expected, atol=1e-3, rtol=0)

if __name__ == '__main__':
    unittest.main()
