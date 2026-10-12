"""
RTEConfig: the deprecated `threshold_std` option warns only when it is passed.
"""

import warnings

import pytest

from pynamicalsys.common.recurrence_quantification_analysis import RTEConfig


def test_defaults_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = RTEConfig()
    assert config.threshold_mode == "std"


@pytest.mark.parametrize("mode", ["direct", "std", "rr"])
def test_explicit_mode_does_not_warn(mode):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = RTEConfig(threshold_mode=mode)
    assert config.threshold_mode == mode


@pytest.mark.parametrize("std, mode", [(True, "std"), (False, "direct")])
def test_threshold_std_alone_warns_and_maps(std, mode):
    with pytest.warns(FutureWarning, match="threshold_std"):
        config = RTEConfig(threshold_std=std)
    assert config.threshold_mode == mode


@pytest.mark.parametrize("std", [True, False])
def test_threshold_std_with_mode_warns_and_is_ignored(std):
    with pytest.warns(FutureWarning, match="ignored"):
        config = RTEConfig(threshold_mode="rr", threshold_std=std)
    assert config.threshold_mode == "rr"
