from distutils.version import LooseVersion

import pytest
import nengo
from nengo.exceptions import BuildError
import numpy as np

from nengo_loihi.decode_neurons import OnOffDecodeNeurons
from nengo_loihi.config import add_params
from nengo_loihi.inputs import (
    ChipReceiveNeurons,
    ChipReceiveNode,
    HostReceiveNode,
    HostSendNode,
)
from nengo_loihi.splitter import (
    PESModulatoryTarget,
    place_ensembles,
    place_internetwork_connections,
    place_nodes,
    place_probes,
    SplitNetworks,
    split,
    split_chip_to_host,
    split_host_neurons_to_chip,
    split_host_to_chip,
    split_host_to_learning_rules,
    split_pre_from_host,
)

default_node_neurons = OnOffDecodeNeurons()


@pytest.mark.parametrize("pre_dims", [1, 3])
@pytest.mark.parametrize("post_dims", [1, 3])
@pytest.mark.parametrize("learn", [True, False])
@pytest.mark.parametrize("use_solver", [True, False])
def test_manual_decoders(
        seed, Simulator, pre_dims, post_dims, learn, use_solver):

    with nengo.Network(seed=seed) as model:
        pre = nengo.Ensemble(50, dimensions=pre_dims,
                             gain=np.ones(50),
                             bias=np.ones(50) * 5)
        post = nengo.Node(size_in=post_dims)

        learning_rule_type = nengo.PES() if learn else None
        weights = np.zeros((post_dims, 50))
        if use_solver:
            conn = nengo.Connection(pre, post,
                                    function=lambda x: np.zeros(post_dims),
                                    learning_rule_type=learning_rule_type,
                                    solver=nengo.solvers.NoSolver(weights.T))
        else:
            conn = nengo.Connection(pre.neurons, post,
                                    learning_rule_type=learning_rule_type,
                                    transform=weights)

        if learn:
            error = nengo.Node(np.zeros(post_dims))
            nengo.Connection(error, conn.learning_rule)

        pre_probe = nengo.Probe(pre.neurons, synapse=None)
        post_probe = nengo.Probe(post, synapse=None)

    if not use_solver and learn:
        with pytest.raises(NotImplementedError):
            with Simulator(model) as sim:
                pass
    else:
        with Simulator(model) as sim:
            sim.run(0.1)

        # Ensure pre population has a lot of activity
        assert np.mean(sim.data[pre_probe]) > 100
        # But that post has no activity due to the zero weights
        assert np.all(sim.data[post_probe] == 0)


def test_place_nodes():
    with nengo.Network() as net:
        offchip1 = nengo.Node(0)
        with nengo.Network():
            offchip2 = nengo.Node(np.sin)
        offchip3 = HostSendNode(dimensions=1)
        onchip = ChipReceiveNode(dimensions=1, size_out=1)

    networks = SplitNetworks(net, node_neurons=default_node_neurons)
    place_nodes(networks)
    assert networks.moves[offchip1] == "host"
    assert networks.moves[offchip2] == "host"
    assert networks.moves[offchip3] == "host"
    assert networks.moves[onchip] == "chip"


def test_place_ensembles():
    with nengo.Network() as net:
        add_params(net)
        offchip = nengo.Ensemble(10, 1, label="offchip")
        net.config[offchip].on_chip = False
        direct = nengo.Ensemble(
            1, 1, neuron_type=nengo.Direct(), label="direct")
        with nengo.Network():
            onchip = nengo.Ensemble(20, 1, label="onchip")
        pre = nengo.Ensemble(10, 1, label="pre")
        post = nengo.Ensemble(10, 1, label="post")
        error = nengo.Ensemble(10, 1, label="error")
        conn = nengo.Connection(pre, post, learning_rule_type=nengo.PES())
        nengo.Connection(error, conn.learning_rule)

    networks = SplitNetworks(net, node_neurons=default_node_neurons)
    place_ensembles(networks)
    assert networks.moves[offchip] == "host"
    assert networks.moves[direct] == "host"
    assert networks.moves[onchip] == "chip"
    assert networks.moves[pre] == "chip"
    assert networks.moves[post] == "host"
    assert networks.moves[error] == "host"


def test_place_inter_network_connection():
    with nengo.Network() as net:
        offchip = nengo.Ensemble(10, 1)
        onchip = nengo.Ensemble(10, 1)
        onon = nengo.Connection(onchip, onchip)
        onoff = nengo.Connection(onchip, offchip)
        offon = nengo.Connection(offchip, onchip)
        offoff = nengo.Connection(offchip, offchip)

    networks = SplitNetworks(net, node_neurons=default_node_neurons)
    networks.move(onchip, "chip")
    networks.move(offchip, "host")

    place_internetwork_connections(networks, networks.original.all_connections)
    assert onoff not in networks
    assert offon not in networks
    assert networks.location(onon) == "chip"
    assert networks.location(offoff) == "host"


def test_split_host_neurons_to_chip():
    with nengo.Network() as net:
        offchip = nengo.Ensemble(10, 1)
        onchip = nengo.Ensemble(10, 1)
        neurons2neurons = nengo.Connection(
            offchip.neurons, onchip.neurons, transform=np.ones((10, 10)))
        neurons2ensemble = nengo.Connection(
            offchip.neurons, onchip, transform=np.ones((1, 10)))

    networks = SplitNetworks(net, node_neurons=default_node_neurons)
    networks.move(offchip, "host")
    networks.move(onchip, "chip")

    def assert_split_correctly(split_conn):
        assert len(networks.adds) == 4
        added_types = sorted([(type(obj).__name__, location)
                              for obj, location in networks.adds.items()])
        assert added_types == [
            ("ChipReceiveNeurons", "chip"),
            ("Connection", "chip"),
            ("Connection", "host"),
            ("HostSendNode", "host"),
        ]
        assert split_conn in networks.removes

        send = next(obj for obj in networks.adds
                    if isinstance(obj, HostSendNode))
        receive = next(obj for obj in networks.adds
                       if isinstance(obj, ChipReceiveNeurons))
        assert networks.host2chip_senders[send] is receive

    split_host_neurons_to_chip(networks, neurons2neurons)
    assert_split_correctly(neurons2neurons)
    networks.adds.clear()  # Makes testing subsequent adds easier
    split_host_neurons_to_chip(networks, neurons2ensemble)
    assert_split_correctly(neurons2ensemble)


def test_split_host_to_chip():
    with nengo.Network() as net:
        ens_offchip = nengo.Ensemble(10, 1)
        node_offchip = nengo.Node(np.sin)
        ens_onchip = nengo.Ensemble(10, 1)
        connections = [
            nengo.Connection(ens_offchip, ens_onchip),
            nengo.Connection(node_offchip, ens_onchip),
            nengo.Connection(
                ens_offchip, ens_onchip.neurons, transform=np.ones((10, 1))),
            nengo.Connection(
                node_offchip, ens_onchip.neurons, transform=np.ones((10, 1))),
        ]

    networks = SplitNetworks(net, node_neurons=default_node_neurons)
    networks.move(ens_offchip, "host")
    networks.move(node_offchip, "host")
    networks.move(ens_onchip, "chip")

    for conn in connections:
        split_host_to_chip(networks, conn)
        for added in networks.adds:
            if isinstance(added, nengo.Ensemble):
                ens = added
            elif isinstance(added, ChipReceiveNode):
                receive = added
            elif isinstance(added, HostSendNode):
                send = added
            # Otherwise must be connection
            elif added.pre is conn.pre:
                pre2ens = added
            elif added.post is conn.post:
                receive2post = added
            else:
                ensneurons2send = added

        assert networks.location(ens) == "host"
        assert isinstance(ens.neuron_type, nengo.SpikingRectifiedLinear)
        assert pre2ens.post is ens

        assert networks.location(receive) == "chip"
        assert networks.location(receive2post) == "chip"
        assert receive2post.pre is receive

        assert networks.location(send) == "host"
        assert networks.location(ensneurons2send) == "host"
        assert ensneurons2send.pre == ens.neurons
        assert ensneurons2send.post is send

        assert conn in networks.removes
        networks.adds.clear()  # makes next loop iteration easier


def test_split_no_node_neuron_error():
    with nengo.Network() as net:
        add_params(net)
        node_offchip = nengo.Node(np.sin)
        ens_onchip = nengo.Ensemble(10, 1)
        nengo.Connection(node_offchip, ens_onchip)

    with pytest.raises(BuildError, match="DecodeNeurons"):
        split(net, precompute=False, node_neurons=None, node_tau=None)


def test_split_chip_to_host():
    with nengo.Network() as net:
        ens_onchip = nengo.Ensemble(10, 1)
        ens_offchip = nengo.Ensemble(10, 1)
        node_offchip = nengo.Node(size_in=1)
        connections = [
            nengo.Connection(ens_onchip, ens_offchip),
            nengo.Connection(
                ens_onchip, ens_offchip, learning_rule_type=nengo.PES()),
            nengo.Connection(ens_onchip, node_offchip),
            nengo.Connection(
                ens_onchip.neurons, ens_offchip, transform=np.ones((1, 10))),
            nengo.Connection(
                ens_onchip.neurons, node_offchip, transform=np.ones((1, 10))),
        ]
        connections.append(
            nengo.Connection(ens_onchip, connections[1].learning_rule)
        )

    networks = SplitNetworks(net, node_neurons=default_node_neurons)
    networks.move(ens_onchip, "chip")
    networks.move(ens_offchip, "host")
    networks.move(node_offchip, "host")

    for conn in connections:
        split_chip_to_host(networks, conn)
        for added in networks.adds:
            if isinstance(added, HostReceiveNode):
                receive = added
            elif isinstance(added, nengo.Probe):
                probe = added
            else:
                assert added.post is conn.post
                receive2post = added

        assert networks.location(receive) == "host"
        assert networks.location(receive2post) == "host"
        assert receive2post.pre is receive

        assert networks.location(probe) == "chip"
        assert probe.target is conn.pre or probe.target is conn.pre.ensemble
        assert probe.synapse is None
        assert probe in networks.chip2host_params
        assert probe in networks.chip2host_receivers
        assert networks.chip2host_receivers[probe] is receive
        if conn.learning_rule_type is not None:
            assert conn.learning_rule in networks.needs_sender
            assert isinstance(networks.needs_sender[conn.learning_rule],
                              PESModulatoryTarget)

        assert conn in networks.removes
        networks.adds.clear()  # makes next loop iteration easier


def test_split_host_to_learning_rule():
    with nengo.Network() as net:
        pre = nengo.Ensemble(10, 1, label="pre")
        post = nengo.Ensemble(10, 1, label="post")
        err_onchip = nengo.Ensemble(10, 1, label="err_onchip")
        err_offchip = nengo.Ensemble(10, 1, label="err_offchip")
        ens_conn = nengo.Connection(pre, post, learning_rule_type=nengo.PES())
        neurons_conn = nengo.Connection(pre.neurons, post.neurons,
                                        learning_rule_type=nengo.PES())
        on2on_ens = nengo.Connection(err_onchip, ens_conn.learning_rule)
        on2on_neurons = nengo.Connection(
            err_onchip, neurons_conn.learning_rule)
        off2on_ens = nengo.Connection(err_offchip, ens_conn.learning_rule)
        off2on_neurons = nengo.Connection(
            err_offchip, neurons_conn.learning_rule)

    networks = SplitNetworks(net, node_neurons=default_node_neurons)
    networks.move(pre, "chip")
    networks.move(post, "chip")
    networks.move(err_onchip, "chip")
    networks.move(err_offchip, "host")
    networks.move(ens_conn, "chip")
    networks.move(neurons_conn, "chip")
    networks.needs_sender[ens_conn.learning_rule] = "ens_pes_target"
    networks.needs_sender[neurons_conn.learning_rule] = "neurons_pes_target"

    split_host_to_learning_rules(networks, networks.original.all_connections)
    assert on2on_ens not in networks
    assert on2on_neurons not in networks
    assert sorted([type(obj).__name__ for obj in networks.adds]) == [
        "Connection", "Connection", "HostSendNode", "HostSendNode",
    ]
    assert off2on_ens in networks.removes
    assert "ens_pes_target" in list(networks.host2chip_senders.values())
    assert off2on_neurons in networks.removes
    assert "neurons_pes_target" in list(networks.host2chip_senders.values())


def test_place_probes():
    with nengo.Network() as net:
        offchip1 = nengo.Node(0)
        with nengo.Network():
            onchip1 = nengo.Ensemble(10, 1)
            offchip2 = nengo.Ensemble(10, 1)
        onchip2 = nengo.Ensemble(10, 1)
        onchip3 = nengo.Connection(onchip1, onchip2)
        offchip3 = nengo.Connection(offchip1, offchip2)
        offchip_probes = [
            nengo.Probe(offchip1),
            nengo.Probe(offchip2),
            nengo.Probe(offchip3),
        ]
        onchip_probes = [
            nengo.Probe(onchip1),
            nengo.Probe(onchip2),
            nengo.Probe(onchip3),
        ]

    networks = SplitNetworks(net, node_neurons=default_node_neurons)
    for obj in [offchip1, offchip2, offchip3]:
        networks.move(obj, "host")
    for obj in [onchip1, onchip2, onchip3]:
        networks.move(obj, "chip")
    place_probes(networks)
    assert all(networks.location(p) == "host" for p in offchip_probes)
    assert all(networks.location(p) == "chip" for p in onchip_probes)


def test_split_pre_from_host():
    with nengo.Network() as net:
        pre_1 = nengo.Node(0, label="pre_1")
        pre_2 = nengo.Ensemble(10, 1, label="pre_2")
        pre_3 = nengo.Node(size_in=1, label="pre_3")
        pre_4 = nengo.Ensemble(1, 1, label="pre_4")
        send = HostSendNode(dimensions=1)
        onchip = nengo.Ensemble(1, 1, label="onchip")
        post1 = nengo.Ensemble(10, 1, label="post1")
        post2 = nengo.Node(size_in=1, label="post2")
        pre_connections = [
            nengo.Connection(pre_1, pre_2),
            nengo.Connection(pre_2, pre_3),
            nengo.Connection(pre_3, pre_4),
            nengo.Connection(pre_4.neurons, send),
        ]
        post_connections = [
            nengo.Connection(onchip, post1),
            nengo.Connection(post1, post2),
        ]

    networks = SplitNetworks(net, node_neurons=default_node_neurons)
    for obj in [pre_1, pre_3, send, post1, post2]:
        networks.move(obj, "host")
    for obj in [pre_2, pre_4]:
        networks.add(obj, "host")
    for conn in pre_connections + post_connections:
        networks.move(conn, "host")
    networks.move(onchip, "chip")

    split_pre_from_host(networks)
    for obj in [pre_1, pre_2, pre_3, pre_4, send] + pre_connections:
        assert networks.location(obj) == "host_pre", obj
    for obj in [post1, post2] + post_connections:
        assert networks.location(obj) == "host", obj
    assert networks.location(onchip) == "chip"


def test_consistent_order():
    with nengo.Network() as model:
        add_params(model)

        u0 = nengo.Node(0, label="u0")
        for i in range(5):
            e = nengo.Ensemble(i+1, 1, label="e%d" % i)
            f = nengo.Ensemble(i+1, 1, label="f%d" % i)
            nengo.Connection(u0, e, label="c0%d" % i)
            nengo.Connection(e, f, label="cf%d" % i)
            nengo.Probe(e)
            nengo.Probe(f.neurons)

    # Test splitting a number of times, making sure the order of things matches
    # the original network each time
    split_params = dict(
        precompute=False,
        node_neurons=OnOffDecodeNeurons(dt=0.001),
        node_tau=0.005,
        remove_passthrough=False,
    )

    networks0 = split(model, **split_params)
    for _ in range(5):
        networks = split(model, **split_params)

        # --- order matches original network
        assert len(model.all_ensembles) == len(networks.chip.all_ensembles)
        for ea, eb in zip(model.all_ensembles, networks.chip.all_ensembles):
            assert ea.n_neurons == eb.n_neurons and ea.label == eb.label

        # --- order matches previous split
        for attr in ('connections', 'ensembles', 'nodes', 'probes'):
            for net in ('host_pre', 'host', 'chip'):
                aa = getattr(getattr(networks0, net), 'all_' + attr)
                bb = getattr(getattr(networks, net), 'all_' + attr)
                for a, b in zip(aa, bb):
                    assert a.label == b.label


@pytest.mark.skipif(LooseVersion(nengo.__version__) <= LooseVersion('2.8.0'),
                    reason="requires more recent Nengo version")
def test_split_conv2d_transform_error():
    with nengo.Network() as net:
        add_params(net)
        node_offchip = nengo.Node([1])
        ens_onchip = nengo.Ensemble(10, 1)
        conv2d = nengo.Convolution(
            n_filters=1, input_shape=(1, 1, 1), kernel_size=(1, 1))
        nengo.Connection(node_offchip, ens_onchip, transform=conv2d)

    with pytest.raises(BuildError, match="Conv2D"):
        split(net, precompute=False, node_neurons=default_node_neurons,
              node_tau=0.005)


def test_split_precompute_loop_error():
    with nengo.Network() as net:
        add_params(net)
        node_offchip = nengo.Node(lambda t, x: x + 1, size_in=1, size_out=1)
        ens_onchip = nengo.Ensemble(10, 1)
        nengo.Connection(node_offchip, ens_onchip)
        nengo.Connection(ens_onchip, node_offchip)

    with pytest.raises(BuildError, match="precompute"):
        split(net, precompute=True, node_neurons=default_node_neurons,
              node_tau=0.005)


def test_splitnetwork_bad_add_type():
    net = nengo.Network()
    networks = SplitNetworks(net)
    networks.add(1, "chip")
    with pytest.raises(AssertionError):
        networks.finalize()


def test_splitnetwork_remove_add():
    net = nengo.Network()
    networks = SplitNetworks(net)
    e = nengo.Ensemble(1, 1, add_to_container=False)
    networks.add(e, "chip")
    networks.remove(e)
    assert e not in networks.adds


def test_pesmodulatorytarget_interface():
    target = "target"
    p = PESModulatoryTarget(target)

    t0 = 4
    e0 = [1.8, 2.4, 3.3]
    t1 = t0 + 3
    e1 = [7.2, 2.2, 4.1]
    e01 = np.array(e0) + np.array(e1)

    p.receive(t0, e0)
    assert isinstance(p.errors[t0], np.ndarray)
    assert np.allclose(p.errors[t0], e0)

    p.receive(t0, e1)
    assert np.allclose(p.errors[t0], e01)

    with pytest.raises(AssertionError):
        p.receive(t0 - 1, e0)  # time needs to be >= last time

    p.receive(t1, e1)
    assert np.allclose(p.errors[t1], e1)

    errors = list(p.collect_errors())
    assert len(errors) == 2
    assert errors[0][:2] == (target, t0) and np.allclose(errors[0][2], e01)
    assert errors[1][:2] == (target, t1) and np.allclose(errors[1][2], e1)

    p.clear()
    assert len(list(p.collect_errors())) == 0


def test_bad_obj_type():
    split = SplitNetworks(nengo.Network())
    split.adds = {"woops": "host"}
    with pytest.raises(AssertionError, match="cannot handle type"):
        split.finalize()
