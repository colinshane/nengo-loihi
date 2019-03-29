from collections import defaultdict, OrderedDict
import logging

from nengo import Network, Node, Ensemble, Connection, Probe
from nengo.builder import Model as NengoModel
from nengo.builder.builder import Builder as NengoBuilder
from nengo.builder.network import build_network
from nengo.cache import NoDecoderCache

from nengo_loihi.block import LoihiBlock
from nengo_loihi.builder.inputs import LoihiInput
from nengo_loihi.decode_neurons import (
    Preset10DecodeNeurons,
    OnOffDecodeNeurons,
)

logger = logging.getLogger(__name__)


class Model:
    """The data structure for the emulator/hardware simulator.

    Defines methods for adding inputs and blocks. Also handles build
    functions, and information associated with building the Nengo model.

    Parameters
    ----------
    dt : float, optional (Default: 0.001)
        The length of a simulator timestep, in seconds.
    label : str, optional (Default: None)
        A name or description to differentiate models.
    builder : Builder, optional (Default: None)
        A `.Builder` instance to keep track of build functions.
        If None, the default builder will be used.

    Attributes
    ----------
    builder : Builder
        The build functions used by this model.
    dt : float
        The length of a simulator timestep, in seconds.
    chip2host_params : dict
        Mapping from Nengo objects to any additional parameters associated
        with those objects for use during the build process.
    decode_neurons : DecodeNeurons
        Type of neurons used to facilitate decoded (NEF-style) connections.
    decode_tau : float
        Time constant of lowpass synaptic filter used with decode neurons.
    blocks : list of LoihiBlock
        List of Loihi blocks simulated by this model.
    inputs : list of LoihiInput
        List of inputs to this model.
    intercept_limit : float
        Limit for clipping intercepts, to avoid neurons with high gains.
    label : str or None
        A name or description to differentiate models.
    node_neurons : DecodeNeurons
        Type of neurons used to convert real-valued node outputs to spikes
        for the chip.
    objs : dict
        Dictionary mapping from Nengo objects to Nengo Loihi objects.
    params : dict
        Mapping from objects to namedtuples containing parameters generated
        in the build process.
    pes_error_scale : float
        Scaling for PES errors, before rounding and clipping to -127..127.
    pes_wgt_exp : int
        Learning weight exponent (base 2) for PES learning connections. This
        controls the maximum weight magnitude (where a larger exponent means
        larger potential weights, but lower weight resolution).
    probes : list
        List of all probes. Probes must be added to this list in the build
        process, as this list is used by Simulator.
    seeded : dict
        All objects are assigned a seed, whether the user defined the seed
        or it was automatically generated. 'seeded' keeps track of whether
        the seed is user-defined. We consider the seed to be user-defined
        if it was set directly on the object, or if a seed was set on the
        network in which the object resides, or if a seed was set on any
        ancestor network of the network in which the object resides.
    seeds : dict
        Mapping from objects to the integer seed assigned to that object.
    vth_nonspiking : float
        Voltage threshold for non-spiking neurons (i.e. voltage decoders).
    """
    def __init__(self, dt=0.001, label=None, builder=None):
        self.dt = dt
        self.label = label
        self.builder = Builder() if builder is None else builder
        self.build_callback = None
        self.decoder_cache = NoDecoderCache()

        # Host models filled in by the build process
        def create_host_model(label, dt):
            # We don't use a decoder cache because it requires a context
            # manager that differs depending on which sub-model is being built
            return NengoModel(
                dt=float(dt),
                label="%s, dt=%f" % (label, dt),
                decoder_cache=NoDecoderCache())

        # TODO: these models may not look/behave exactly the same as
        # standard nengo models, because they don't have a toplevel network
        # built into them or configs set
        self.host_pre = create_host_model(label="%s, host_pre" % label, dt=dt)
        self.host = create_host_model(label="%s, host" % label, dt=dt)

        # Objects created by the model for simulation on Loihi
        self.inputs = OrderedDict()
        self.blocks = OrderedDict()

        # Will be filled in by the simulator __init__
        self.splitter_directive = None

        # Will be filled in by the network builder
        self.toplevel = None
        self.config = None

        # Resources used by the build process
        self.objs = defaultdict(dict)
        self.params = {}  # Holds data generated when building objects
        self.probes = []
        self.probe_conns = {}
        self.seeds = {}
        self.seeded = {}

        # --- other (typically standard) parameters
        # Filter on decode neurons
        self.decode_tau = 0.005
        # ^TODO: how to choose this filter? Even though the input is spikes,
        # it may not be absolutely necessary since tau_rc provides a filter,
        # and maybe we don't want double filtering if connection has a filter

        self.decode_neurons = Preset10DecodeNeurons(dt=dt)
        self.node_neurons = OnOffDecodeNeurons(dt=dt)

        # voltage threshold for non-spiking neurons (i.e. voltage decoders)
        self.vth_nonspiking = 10

        # limit for clipping intercepts, to avoid neurons with high gains
        self.intercept_limit = 0.95

        # scaling for PES errors, before rounding and clipping to -127..127
        self.pes_error_scale = 100.

        # learning weight exponent for PES (controls the maximum weight
        # magnitude/weight resolution)
        self.pes_wgt_exp = 4

        # Used to track interactions between host models
        self.chip2host_params = {}
        self.chip2host_receivers = OrderedDict()
        self.host2chip_senders = OrderedDict()
        self.needs_sender = {}

    def __getstate__(self):
        raise NotImplementedError("Can't pickle nengo_loihi.builder.Model")

    def __setstate__(self, state):
        raise NotImplementedError("Can't pickle nengo_loihi.builder.Model")

    def __str__(self):
        return "%s(%s)" % (type(self).__name__, self.label)

    def add_input(self, input):
        assert isinstance(input, LoihiInput)
        assert input not in self.inputs
        self.inputs[input] = len(self.inputs)

    def add_block(self, block):
        assert isinstance(block, LoihiBlock)
        assert block not in self.blocks
        self.blocks[block] = len(self.blocks)

    def delegate(self, obj):
        if not isinstance(obj, (Node, Ensemble, Probe)):
            # Note: this is safe because any objects built from within a normal
            # nengo model (other than self) will not be re-delegated
            return self
        elif self.splitter_directive.on_chip(obj):
            return self
        elif self.splitter_directive.is_precomputable(obj):
            return self.host_pre
        else:
            return self.host

    def build(self, obj, *args, **kwargs):
        # Don't build the passthrough nodes or connections
        passthrough_directive = self.splitter_directive.passthrough_directive
        if (isinstance(obj, Node)
                and obj in passthrough_directive.removed_passthroughs):
            return None
        if (isinstance(obj, Connection)
                and obj in passthrough_directive.removed_connections):
            return None

        # Note: any callbacks for host_pre or host will not be invoked here
        model = self.delegate(obj)
        if model is not self:
            # done for compatibility with nengo<=2.8.0
            # otherwise we could just copy over the initial
            # seeding to all other models
            model.seeds[obj] = self.seeds[obj]
            model.seeded[obj] = self.seeded[obj]

        built = model.builder.build(model, obj, *args, **kwargs)
        if self.build_callback is not None:
            self.build_callback(obj)
        return built

    def has_built(self, obj):
        return obj in self.params


class Builder(NengoBuilder):
    """Fills in the Loihi Model object based on the Nengo Network.

    We cannot use the Nengo builder as is because we make normal Nengo
    networks for host-to-chip and chip-to-host communication. To keep
    Nengo and Nengo Loihi builders separate, we make a blank subclass,
    which effectively copies the class.
    """

    builders = {}


Builder.register(Network)(build_network)
