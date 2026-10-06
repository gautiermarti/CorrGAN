import numpy as np
import pytest

from corrgan.legacy import load_published_ccorrgan, nhwc_projection_to_nchw


def test_dense_projection_preserves_spatial_and_channel_locations():
    rng = np.random.default_rng(17)
    kernel = rng.normal(size=(5, 2 * 3 * 4))
    bias = rng.normal(size=24)
    noise = rng.normal(size=(7, 5))
    expected = (noise @ kernel + bias).reshape(7, 2, 3, 4).transpose(0, 3, 1, 2)
    converted, offset = nhwc_projection_to_nchw(
        kernel, bias, height=2, width=3, channels=4
    )
    actual = (noise @ converted.T + offset).reshape(7, 4, 2, 3)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_rejects_unrecognized_checkpoint_before_hdf5_parsing(tmp_path):
    path = tmp_path / "wrong.h5"
    path.write_bytes(b"not the published checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        load_published_ccorrgan(path)
