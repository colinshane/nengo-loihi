import inspect

import nengo
from nengo.exceptions import ReadonlyError, ValidationError, SimulationError
import numpy as np
import pytest

import nengo_loihi
from nengo_loihi.block import Axon, LoihiBlock, Probe, Synapse
from nengo_loihi.builder import Model
from nengo_loihi.discretize import discretize_model
from nengo_loihi.emulator import EmulatorInterface
from nengo_loihi.hardware import HardwareInterface
from nengo_loihi.hardware.allocators import RoundRobin
from nengo_loihi.inputs import SpikeInput


def test_model_validate_notempty(Simulator):
    with nengo.Network() as model:
        nengo_loihi.add_params(model)

        a = nengo.Ensemble(10, 1)
        model.config[a].on_chip = False

    with pytest.raises(nengo.exceptions.BuildError):
        with Simulator(model):
            pass


@pytest.mark.parametrize("precompute", [True, False])
def test_probedict_fallbacks(precompute, Simulator):
    with nengo.Network() as net:
        nengo_loihi.add_params(net)
        node_a = nengo.Node(0)
        with nengo.Network():
            ens_b = nengo.Ensemble(10, 1)
            conn_ab = nengo.Connection(node_a, ens_b)
        ens_c = nengo.Ensemble(5, 1)
        net.config[ens_c].on_chip = False
        conn_bc = nengo.Connection(ens_b, ens_c)
        probe_a = nengo.Probe(node_a)
        probe_c = nengo.Probe(ens_c)

    with Simulator(net, precompute=precompute) as sim:
        sim.run(0.002)

    assert node_a in sim.data
    assert ens_b in sim.data
    assert ens_c in sim.data
    assert probe_a in sim.data
    assert probe_c in sim.data

    # TODO: connections are currently not probeable as they are
    #       replaced in the splitting process
    assert conn_ab  # in sim.data
    assert conn_bc  # in sim.data


def test_probedict_interface(Simulator):
    with nengo.Network(label='net') as net:
        u = nengo.Node(1, label='u')
        a = nengo.Ensemble(9, 1, label='a')
        nengo.Connection(u, a)

    with Simulator(net) as sim:
        pass

    objs = [u, a]
    count = 0
    for o in sim.data:
        count += 1
        if o in objs:
            objs.remove(o)
    assert len(sim.data) == count
    assert len(objs) == 0, "Objects did not appear in probedict: %s" % objs


@pytest.mark.xfail
@pytest.mark.parametrize(
    "dt, pre_on_chip",
    [(2e-4, True), (3e-4, False), (4e-4, True), (2e-3, True)]
)
def test_dt(dt, pre_on_chip, Simulator, seed, plt, allclose):
    function = lambda x: x**2
    probe_synapse = nengo.Alpha(0.01)
    simtime = 0.2

    ens_params = dict(
        intercepts=nengo.dists.Uniform(-0.9, 0.9),
        max_rates=nengo.dists.Uniform(100, 120))

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(lambda t: -np.sin(2 * np.pi * t / simtime))
        stim_p = nengo.Probe(stim, synapse=probe_synapse)

        pre = nengo.Ensemble(100, 1, **ens_params)
        model.config[pre].on_chip = pre_on_chip
        pre_p = nengo.Probe(pre, synapse=probe_synapse)

        post = nengo.Ensemble(101, 1, **ens_params)
        post_p = nengo.Probe(post, synapse=probe_synapse)

        nengo.Connection(stim, pre)
        nengo.Connection(pre, post, function=function,
                         solver=nengo.solvers.LstsqL2(weights=True))

    with Simulator(model, dt=dt) as sim:
        sim.run(simtime)

    x = sim.data[stim_p]
    y = function(x)
    plt.plot(sim.trange(), x, 'k--')
    plt.plot(sim.trange(), y, 'k--')
    plt.plot(sim.trange(), sim.data[pre_p])
    plt.plot(sim.trange(), sim.data[post_p])

    assert allclose(sim.data[pre_p], x, rtol=0.1, atol=0.1)
    assert allclose(sim.data[post_p], y, rtol=0.1, atol=0.1)


@pytest.mark.parametrize('simtype', ['simreal', None])
def test_nengo_comm_channel_compare(simtype, Simulator, seed, plt, allclose):
    if simtype == 'simreal':
        Simulator = lambda *args: nengo_loihi.Simulator(
            *args, target='simreal')

    simtime = 0.6

    with nengo.Network(seed=seed) as model:
        u = nengo.Node(lambda t: np.sin(6*t / simtime))
        a = nengo.Ensemble(50, 1)
        b = nengo.Ensemble(50, 1)
        nengo.Connection(u, a)
        nengo.Connection(a, b, function=lambda x: x**2,
                         solver=nengo.solvers.LstsqL2(weights=True))

        ap = nengo.Probe(a, synapse=0.03)
        bp = nengo.Probe(b, synapse=0.03)

    with nengo.Simulator(model) as nengo_sim:
        nengo_sim.run(simtime)

    with Simulator(model) as loihi_sim:
        loihi_sim.run(simtime)

    plt.subplot(2, 1, 1)
    plt.plot(nengo_sim.trange(), nengo_sim.data[ap])
    plt.plot(loihi_sim.trange(), loihi_sim.data[ap])

    plt.subplot(2, 1, 2)
    plt.plot(nengo_sim.trange(), nengo_sim.data[bp])
    plt.plot(loihi_sim.trange(), loihi_sim.data[bp])

    assert allclose(loihi_sim.data[ap], nengo_sim.data[ap], atol=0.1, rtol=0.2)
    assert allclose(loihi_sim.data[bp], nengo_sim.data[bp], atol=0.1, rtol=0.2)


@pytest.mark.parametrize("precompute", (True, False))
def test_close(Simulator, precompute):
    with nengo.Network() as net:
        a = nengo.Node([0])
        b = nengo.Ensemble(10, 1)
        c = nengo.Node(size_in=1)
        nengo.Connection(a, b)
        nengo.Connection(b, c)

    with Simulator(net, precompute=precompute) as sim:
        pass

    assert sim.closed
    assert all(s.closed for s in sim.sims.values())


def test_all_run_steps(Simulator):
    # Case 1. No objects on host, so no host and no host_pre
    with nengo.Network() as net:
        pre = nengo.Ensemble(10, 1)
        post = nengo.Ensemble(10, 1)
        nengo.Connection(pre, post)

    # 1a. precompute=False, no host
    with Simulator(net) as sim:
        sim.run(0.001)
    # Since no objects on host, we should be precomputing even if we did not
    # explicitly request precomputing
    assert sim.precompute
    assert inspect.ismethod(sim._run_steps)
    assert sim._run_steps.__name__ == "run_steps"

    # 1b. precompute=True, no host, no host_pre
    with pytest.warns(UserWarning) as record:
        with Simulator(net, precompute=True) as sim:
            sim.run(0.001)
    assert any("No precomputable objects" in r.message.args[0] for r in record)
    assert inspect.ismethod(sim._run_steps)
    assert sim._run_steps.__name__ == "run_steps"

    # Case 2: Add a precomputable off-chip object, so we have either host or
    # host_pre but not both host and host_pre
    with net:
        stim = nengo.Node(1)
        stim_conn = nengo.Connection(stim, pre)

    # 2a. precompute=False, host
    with Simulator(net) as sim:
        sim.run(0.001)
    assert sim._run_steps.__name__.endswith("_bidirectional_with_host")

    # 2b. precompute=True, no host, host_pre
    with Simulator(net, precompute=True) as sim:
        sim.run(0.001)
    assert sim._run_steps.__name__.endswith("_precomputed_host_pre_only")

    # Case 3: Add a non-precomputable off-chip object so we have host
    # and host_pre
    with net:
        out = nengo.Node(size_in=1)
        nengo.Connection(post, out)
        nengo.Probe(out)  # probe to prevent `out` from being optimized away

    # 3a. precompute=False, host (same as 2a)
    with Simulator(net) as sim:
        sim.run(0.001)
    assert sim._run_steps.__name__.endswith("_bidirectional_with_host")

    # 3b. precompute=True, host, host_pre
    with Simulator(net, precompute=True) as sim:
        sim.run(0.001)
    assert sim._run_steps.__name__.endswith("_precomputed_host_pre_and_host")

    # Case 4: Delete the precomputable off-chip object, so we have host only
    net.nodes.remove(stim)
    net.connections.remove(stim_conn)

    # 4a. precompute=False, host (same as 2a and 3a)
    with Simulator(net) as sim:
        sim.run(0.001)
    assert sim._run_steps.__name__.endswith("_bidirectional_with_host")

    # 4b. precompute=True, host, no host_pre
    with pytest.warns(UserWarning) as record:
        with Simulator(net, precompute=True) as sim:
            sim.run(0.001)
    assert any("No precomputable objects" in r.message.args[0] for r in record)
    assert sim._run_steps.__name__.endswith("_precomputed_host_only")


def test_no_precomputable(Simulator):
    with nengo.Network() as net:
        active_ens = nengo.Ensemble(10, 1,
                                    gain=np.ones(10) * 10,
                                    bias=np.ones(10) * 10)
        out = nengo.Node(size_in=10)
        nengo.Connection(active_ens.neurons, out)
        out_p = nengo.Probe(out)

    with pytest.warns(UserWarning) as record:
        with Simulator(net, precompute=True) as sim:
            sim.run(0.01)

    assert sim._run_steps.__name__.endswith("precomputed_host_only")
    # Should warn that no objects are precomputable
    assert any("No precomputable objects" in r.message.args[0] for r in record)
    # But still mark the sim as precomputable for speed reasons, because
    # there are no inputs that depend on outputs in this case
    assert sim.precompute
    assert sim.data[out_p].shape[0] == sim.trange().shape[0]
    assert np.all(sim.data[out_p][-1] > 100)


def test_all_onchip(Simulator):
    with nengo.Network() as net:
        active_ens = nengo.Ensemble(10, 1,
                                    gain=np.ones(10) * 10,
                                    bias=np.ones(10) * 10)
        out = nengo.Ensemble(10, 1, gain=np.ones(10), bias=np.ones(10))
        nengo.Connection(active_ens.neurons, out.neurons,
                         transform=np.eye(10) * 10)
        out_p = nengo.Probe(out.neurons)

    with Simulator(net) as sim:
        sim.run(0.01)

    # Though we did not specify precompute, the model should be marked as
    # precomputable because there are no off-chip objects
    assert sim.precompute
    assert inspect.ismethod(sim._run_steps)
    assert sim._run_steps.__name__ == "run_steps"
    assert sim.data[out_p].shape[0] == sim.trange().shape[0]
    assert np.all(sim.data[out_p][-1] > 100)


@pytest.mark.skipif(pytest.config.getoption('--target') != 'loihi',
                    reason="snips only on Loihi")
def test_snips_round_robin_unsupported(Simulator):
    with nengo.Network() as model:
        # input is required otherwise precompute will be
        # automatically overwritten to True (and then no snips)
        u = nengo.Node(0)
        x = nengo.Ensemble(1, 1)
        nengo.Connection(u, x)

    with pytest.raises(SimulationError, match="snips are not supported"):
        with Simulator(model, precompute=False,
                       hardware_options={'allocator': RoundRobin(n_chips=8)}):
            pass


def test_progressbar_values(Simulator):
    with nengo.Network() as model:
        nengo.Ensemble(1, 1)

    # both `None` and `False` are valid ways of specifying no progress bar
    with Simulator(model, progress_bar=None):
        pass

    with Simulator(model, progress_bar=False):
        pass

    # progress bar not yet implemented
    with pytest.warns(UserWarning, match="progress bar"):
        with Simulator(model, progress_bar=True):
            pass


def test_tau_s_warning(Simulator):
    with nengo.Network() as net:
        stim = nengo.Node(0)
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(stim, ens, synapse=0.1)
        nengo.Connection(ens, ens,
                         synapse=0.001,
                         solver=nengo.solvers.LstsqL2(weights=True))

    with pytest.warns(UserWarning) as record:
        with Simulator(net):
            pass
    # The 0.001 synapse is applied first due to splitting rules putting
    # the stim -> ens connection later than the ens -> ens connection
    assert any(rec.message.args[0] == (
        "tau_s is currently 0.001, which is smaller than 0.005. "
        "Overwriting tau_s with 0.005.") for rec in record)

    with net:
        nengo.Connection(ens, ens,
                         synapse=0.1,
                         solver=nengo.solvers.LstsqL2(weights=True))
    with pytest.warns(UserWarning) as record:
        with Simulator(net):
            pass
    assert any(rec.message.args[0] == (
        "tau_s is already set to 0.1, which is larger than 0.005. Using 0.1."
    ) for rec in record)


@pytest.mark.xfail(nengo.version.version_info <= (2, 8, 0),
                   reason="Nengo core controls seeds")
@pytest.mark.parametrize('precompute', [False, True])
def test_seeds(precompute, Simulator, seed):
    with nengo.Network(seed=seed) as net:
        nengo_loihi.add_params(net)

        e0 = nengo.Ensemble(1, 1)
        e1 = nengo.Ensemble(1, 1, seed=2)
        e2 = nengo.Ensemble(1, 1)
        net.config[e2].on_chip = False
        nengo.Connection(e0, e1)
        nengo.Connection(e0, e2)

        with nengo.Network():
            n = nengo.Node(0)
            e = nengo.Ensemble(1, 1)
            nengo.Node(1)
            nengo.Connection(n, e)
            nengo.Probe(e)

        with nengo.Network(seed=8):
            nengo.Ensemble(8, 1, seed=3)
            nengo.Node(1)

    # --- test that seeds are the same as nengo ref simulator
    ref = nengo.Simulator(net)

    with Simulator(net, precompute=precompute) as sim:
        for obj in net.all_objects:
            on_chip = (not isinstance(obj, nengo.Node) and (
                not isinstance(obj, nengo.Ensemble)
                or net.config[obj].on_chip))

            seed = sim.model.seeds.get(obj, None)
            assert seed is None or seed == ref.model.seeds[obj]
            if on_chip:
                assert seed is not None
            if obj in sim.model.seeded:
                assert sim.model.seeded[obj] == ref.model.seeded[obj]

            if precompute:
                seed0 = sim.sims["host_pre"].model.seeds.get(obj, None)
                assert seed0 is None or seed0 == ref.model.seeds[obj]
                seed1 = sim.sims["host"].model.seeds.get(obj, None)
                assert seed1 is None or seed1 == ref.model.seeds[obj]
            else:
                seed0 = sim.sims["host"].model.seeds.get(obj, None)
                assert seed0 is None or seed0 == ref.model.seeds[obj]
                seed1 = None

            if not on_chip:
                assert seed0 is not None or seed1 is not None

    # --- test that seeds that we set are preserved after splitting
    model = nengo_loihi.builder.Model()
    for i, o in enumerate(net.all_objects):
        model.seeds[o] = i

    with Simulator(net, model=model, precompute=precompute) as sim:
        for i, o in enumerate(net.all_objects):
            for name, subsim in sim.sims.items():
                if name.startswith("host"):
                    assert subsim.model.seeds[o] == i


def test_interface(Simulator, allclose):
    """Tests for the Simulator API for things that aren't covered elsewhere"""
    # test sim.time
    with nengo.Network() as model:
        nengo.Ensemble(2, 1)

    simtime = 0.003
    with Simulator(model) as sim:
        sim.run(simtime)

    assert allclose(sim.time, simtime)

    # test that sim.dt is read-only
    with pytest.raises(ReadonlyError, match="dt"):
        sim.dt = 0.002

    # test error for bad target
    with pytest.raises(ValidationError, match="target"):
        with Simulator(model, target="foo"):
            pass

    # test negative runtime
    with pytest.raises(ValidationError, match="[Mm]ust be positive"):
        with Simulator(model):
            sim.run(-0.1)

    # test zero step warning
    with pytest.warns(UserWarning, match="0 timesteps"):
        with Simulator(model):
            sim.run(1e-8)


@pytest.mark.hang
@pytest.mark.skipif(pytest.config.getoption('--target') != 'loihi',
                    reason="Only Loihi has special shutdown procedure")
def test_loihi_simulation_exception(Simulator):
    """Test that Loihi shuts down properly after exception during simulation"""
    def node_fn(t):
        if t < 0.002:
            return 0
        else:
            raise RuntimeError("exception to kill the simulation")

    with nengo.Network() as net:
        u = nengo.Node(node_fn)
        e = nengo.Ensemble(8, 1)
        nengo.Connection(u, e)

    with Simulator(net, precompute=False) as sim:
        sim.run(0.01)
        assert not sim.sims['loihi'].nxDriver.conn


@pytest.mark.parametrize('precompute', [True, False])
def test_double_run(precompute, Simulator, seed, allclose):
    simtime = 0.2
    with nengo.Network(seed=seed) as net:
        stim = nengo.Node(lambda t: np.sin((2*np.pi/simtime) * t))
        ens = nengo.Ensemble(10, 1)
        probe = nengo.Probe(ens)
        nengo.Connection(stim, ens, synapse=None)

    with Simulator(net, precompute=True) as sim0:
        sim0.run(simtime)

    with Simulator(net, precompute=precompute) as sim1:
        sim1.run(simtime / 2)
        sim1.run(simtime / 2)

    assert allclose(sim1.time, sim0.time)
    assert len(sim1.trange()) == len(sim0.trange())
    assert allclose(sim1.data[probe], sim0.data[probe])


# These base-10 exp values translate to noiseExp of [5, 10, 13] on the chip.
@pytest.mark.parametrize('exp', [-4.5, -3, -2])
def test_simulator_noise(exp, request, plt, seed, allclose):
    # TODO: test that the mean falls within a number of standard errors
    # of the expected mean, and that non-zero offsets work correctly.
    # Currently, there is an unexpected negative bias for small noise
    # exponents, apparently because there is a probability of generating
    # the shifted equivalent of -128, whereas with e.g. exp = 7 all the
    # generated numbers fall in [-127, 127].
    offset = 0

    target = request.config.getoption("--target")
    n_cx = 1000

    model = Model()
    block = LoihiBlock(n_cx)
    block.compartment.configure_relu()

    block.compartment.vmin = -1

    block.compartment.enableNoise[:] = 1
    block.compartment.noiseExp0 = exp
    block.compartment.noiseMantOffset0 = offset
    block.compartment.noiseAtDendOrVm = 1

    probe = Probe(target=block, key='voltage')
    block.add_probe(probe)
    model.add_block(block)

    discretize_model(model)
    exp2 = block.compartment.noiseExp0
    offset2 = block.compartment.noiseMantOffset0

    n_steps = 100
    if target == 'loihi':
        with HardwareInterface(model, use_snips=False, seed=seed) as sim:
            sim.run_steps(n_steps)
            y = sim.get_probe_output(probe)
    else:
        with EmulatorInterface(model, seed=seed) as sim:
            sim.run_steps(n_steps)
            y = sim.get_probe_output(probe)

    t = np.arange(1, n_steps+1)
    bias = offset2 * 2.**(exp2 - 1)
    std = 2.**exp2 / np.sqrt(3)  # divide by sqrt(3) for std of uniform -1..1
    rmean = t * bias
    rstd = np.sqrt(t) * std
    rerr = rstd / np.sqrt(n_cx)
    ymean = y.mean(axis=1)
    ystd = y.std(axis=1)
    diffs = np.diff(np.vstack([np.zeros_like(y[0]), y]), axis=0)

    plt.subplot(311)
    plt.hist(diffs.ravel(), bins=256)

    plt.subplot(312)
    plt.plot(rmean, 'k')
    plt.plot(rmean + 3*rerr, 'k--')
    plt.plot(rmean - 3*rerr, 'k--')
    plt.plot(ymean)
    plt.title('mean')

    plt.subplot(313)
    plt.plot(rstd, 'k')
    plt.plot(ystd)
    plt.title('std')

    assert allclose(ystd, rstd, rtol=0.1, atol=1)


def test_population_input(request, allclose):
    target = request.config.getoption("--target")
    dt = 0.001

    n_inputs = 3
    n_axons = 1
    n_cx = 2

    steps = 6
    spike_times_inds = [(1, [0]),
                        (3, [1]),
                        (5, [2])]

    model = Model()

    input = SpikeInput(n_inputs)
    model.add_input(input)
    spikes = [(input, ti, inds) for ti, inds in spike_times_inds]

    input_axon = Axon(n_axons)
    axon_map = np.zeros(n_inputs, dtype=int)
    atoms = np.arange(n_inputs)
    input_axon.set_axon_map(axon_map, atoms)
    input.add_axon(input_axon)

    block = LoihiBlock(n_cx)
    block.compartment.configure_lif(tau_rc=0., tau_ref=0., dt=dt)
    block.compartment.configure_filter(0, dt=dt)
    model.add_block(block)

    synapse = Synapse(n_axons)
    weights = 0.1 * np.array([[[1, 2], [2, 3], [4, 5]]], dtype=float)
    indices = np.array([[[0, 1], [0, 1], [0, 1]]], dtype=int)
    axon_to_weight_map = np.zeros(n_axons, dtype=int)
    cx_bases = np.zeros(n_axons, dtype=int)
    synapse.set_population_weights(
        weights, indices, axon_to_weight_map, cx_bases, pop_type=32)
    block.add_synapse(synapse)
    input_axon.target = synapse

    probe = Probe(target=block, key='voltage')
    block.add_probe(probe)

    discretize_model(model)

    if target == 'loihi':
        with HardwareInterface(model, use_snips=True) as sim:
            sim.run_steps(steps, blocking=False)
            for ti in range(1, steps+1):
                spikes_i = [spike for spike in spikes if spike[1] == ti]
                sim.host2chip(spikes=spikes_i, errors=[])
                sim.chip2host(probes_receivers={})

            y = sim.get_probe_output(probe)
    else:
        for inp, ti, inds in spikes:
            inp.add_spikes(ti, inds)

        with EmulatorInterface(model) as sim:
            sim.run_steps(steps)
            y = sim.get_probe_output(probe)

    vth = block.compartment.vth[0]
    assert (block.compartment.vth == vth).all()
    z = y / vth
    assert allclose(z[[1, 3, 5]], weights[0], atol=4e-2, rtol=0)


@pytest.mark.skipif(pytest.config.getoption("--target") != "loihi",
                    reason="Loihi only test")
def test_precompute(allclose, Simulator, seed, plt):
    simtime = 0.2

    with nengo.Network(seed=seed) as model:
        D = 2
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi / simtime)] * D)

        a = nengo.Ensemble(100, D)

        nengo.Connection(stim, a)

        output = nengo.Node(size_in=D)

        nengo.Connection(a, output)

        p_stim = nengo.Probe(stim, synapse=0.03)
        p_a = nengo.Probe(a, synapse=0.03)
        p_out = nengo.Probe(output, synapse=0.03)

    with Simulator(model, precompute=False) as sim1:
        sim1.run(simtime)
    with Simulator(model, precompute=True) as sim2:
        sim2.run(simtime)

    plt.subplot(2, 1, 1)
    plt.plot(sim1.trange(), sim1.data[p_stim])
    plt.plot(sim1.trange(), sim1.data[p_a])
    plt.plot(sim1.trange(), sim1.data[p_out])
    plt.title('precompute=False')
    plt.subplot(2, 1, 2)
    plt.plot(sim2.trange(), sim2.data[p_stim])
    plt.plot(sim2.trange(), sim2.data[p_a])
    plt.plot(sim2.trange(), sim2.data[p_out])
    plt.title('precompute=True')

    assert np.array_equal(sim1.data[p_stim], sim2.data[p_stim])
    assert allclose(sim1.data[p_a], sim2.data[p_a], atol=0.2)
    assert allclose(sim1.data[p_out], sim2.data[p_out], atol=0.2)


@pytest.mark.skipif(pytest.config.getoption("--target") != "loihi",
                    reason="Loihi only test")
@pytest.mark.xfail(pytest.config.getoption("--target") == "loihi",
                   reason="Fails allclose check")
def test_input_node_precompute(allclose, Simulator, plt):
    simtime = 1.0
    input_fn = lambda t: np.sin(6 * np.pi * t / simtime)
    targets = ["sim", "loihi"]
    x = {}
    u = {}
    v = {}
    for target in targets:
        n = 4
        with nengo.Network(seed=1) as model:
            inp = nengo.Node(input_fn)

            a = nengo.Ensemble(n, 1)
            ap = nengo.Probe(a, synapse=0.01)
            aup = nengo.Probe(a.neurons, 'input')
            avp = nengo.Probe(a.neurons, 'voltage')

            nengo.Connection(inp, a)

        with Simulator(model, precompute=True, target=target) as sim:
            print("Running in {}".format(target))
            sim.run(simtime)

        synapse = nengo.synapses.Lowpass(0.03)
        x[target] = synapse.filt(sim.data[ap])

        u[target] = sim.data[aup][:25]
        u[target] = (
            np.round(u[target] * 1000)
            if str(u[target].dtype).startswith('float') else
            u[target])

        v[target] = sim.data[avp][:25]
        v[target] = (
            np.round(v[target] * 1000)
            if str(v[target].dtype).startswith('float') else
            v[target])

        plt.plot(sim.trange(), x[target], label=target)

    t = sim.trange()
    u = input_fn(t)
    plt.plot(t, u, 'k:', label='input')
    plt.legend(loc='best')

    assert allclose(x['sim'], x['loihi'], atol=0.1, rtol=0.01)
