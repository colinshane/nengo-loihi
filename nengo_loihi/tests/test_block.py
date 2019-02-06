from nengo.exceptions import BuildError
import pytest

from nengo_loihi.block import LoihiBlock


def test_compartment_errors():
    block = LoihiBlock(90)

    # set filter to a very large value so current scaling can't be applied
    with pytest.raises(BuildError, match="[Cc]urrent.*scaling"):
        block.compartment.configure_filter(1e6)

    # set to a value when previously configured to larger value
    block.compartment.configure_filter(0.006)
    with pytest.warns(UserWarning, match="tau_s.*larger"):
        block.compartment.configure_filter(0.004)

    # set to a value when previously configured to smaller value
    block.compartment.configure_filter(0.003)
    with pytest.warns(UserWarning, match="tau_s.*smaller"):
        block.compartment.configure_filter(0.007)

    # set tau_rc to a very large value so voltage scaling can't be applied
    with pytest.raises(BuildError, match="[Vv]oltage.*scaling"):
        block.compartment.configure_lif(tau_rc=1e6)
