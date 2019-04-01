from distutils.version import LooseVersion

import nengo
from nengo.exceptions import BuildError
import pytest


@pytest.mark.skipif(LooseVersion(nengo.__version__) <= LooseVersion('2.8.0'),
                    reason="requires more recent Nengo version")
def test_split_conv2d_transform_error(Simulator):
    with nengo.Network() as net:
        node_offchip = nengo.Node([1])
        ens_onchip = nengo.Ensemble(10, 1)
        conv2d = nengo.Convolution(
            n_filters=1, input_shape=(1, 1, 1), kernel_size=(1, 1))
        nengo.Connection(node_offchip, ens_onchip, transform=conv2d)

    with pytest.raises(BuildError, match="Conv2D"):
        with Simulator(net):
            pass
