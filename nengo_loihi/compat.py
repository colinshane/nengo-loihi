from distutils.version import LooseVersion
import logging



logger = logging.getLogger(__name__)


import nengo
nengo_version = LooseVersion(getattr(nengo, "__version__", "0.0.0"))


if nengo_version > LooseVersion('2.8.0'):
    import nengo.transforms as nengo_transforms

    # Transform class for connection transforms
    def transform_array(transform):
        return transform.init

else:
    nengo_transforms = None

    # array-only connection transforms
    def transform_array(transform):
        return transform
