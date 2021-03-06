from distutils.version import LooseVersion
import os
import shutil
import sys
import tempfile

try:
    import nxsdk
    nxsdk_dir = os.path.realpath(
        os.path.join(os.path.dirname(nxsdk.__file__), "..")
    )
    nxsdk_version = LooseVersion(getattr(nxsdk, "__version__", "0.0.0"))
    HAS_NXSDK = True

    def assert_nxsdk():
        pass

    from nxsdk.graph import graph

    class PatchedGraph(graph.Graph):
        def __init__(self, *args, **kwargs):
            super(PatchedGraph, self).__init__(*args, **kwargs)
            self.nengo_tmp_dirs = []

        def createProcess(self, name, cFilePath, *args, **kwargs):
            # copy the c file to a temporary directory (so that multiple
            # simulations can use the same snip files without running into
            # problems)
            tmp = tempfile.TemporaryDirectory()
            self.nengo_tmp_dirs.append(tmp)

            tmp_path = os.path.join(tmp.name, os.path.basename(cFilePath))
            shutil.copyfile(cFilePath, tmp_path)

            return super(PatchedGraph, self).createProcess(
                name, tmp_path, *args, **kwargs)

    graph.Graph = PatchedGraph
except ImportError:
    HAS_NXSDK = False
    nxsdk_dir = None
    nxsdk_version = None

    exception = sys.exc_info()[1]

    def assert_nxsdk(exception=exception):
        raise exception


if HAS_NXSDK and nxsdk_version < LooseVersion("0.8.0"):  # pragma: no cover
    import nxsdk.arch.n2a.compiler.microcodegen.interface as microcodegen_uci
    from nxsdk.arch.n2a.compiler.tracecfggen.tracecfggen import TraceCfgGen
    from nxsdk.arch.n2a.graph.graph import N2Board
    from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator
    from nxsdk.arch.n2a.graph.probes import N2SpikeProbe
elif HAS_NXSDK:
    import nxsdk.compiler.microcodegen.interface as microcodegen_uci
    from nxsdk.compiler.tracecfggen.tracecfggen import TraceCfgGen
    from nxsdk.graph.nxboard import N2Board
    from nxsdk.graph.nxinputgen import BasicSpikeGenerator
    from nxsdk.graph.nxprobes import N2SpikeProbe
else:
    BasicSpikeGenerator = None
    microcodegen_uci = None
    N2Board = None
    N2SpikeProbe = None
    TraceCfgGen = None
    nxsdk = None
    nxsdk_dir = None
