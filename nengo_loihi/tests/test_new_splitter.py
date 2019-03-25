from distutils.version import LooseVersion

import pytest
import nengo
from nengo.exceptions import BuildError
import numpy as np

from nengo_loihi.config import add_params
from nengo_loihi.new_splitter import SplitterDirective


def test_place_nodes():
    # all nodes go on the host
    # ChipReceiveNodes and HostSendNodes are created later by the builder

    with nengo.Network() as net:
        offchip1 = nengo.Node(0)
        with nengo.Network():
            offchip2 = nengo.Node(np.sin)
            ensemble = nengo.Ensemble(100, 1)
            offchip3 = nengo.Node(size_in=1)
            nengo.Connection(ensemble, offchip3)

    with nengo.Network():
        nowhere = nengo.Node(0)

    splitter_directive = SplitterDirective(net)
    assert not splitter_directive.on_chip(offchip1)
    assert not splitter_directive.on_chip(offchip2)
    assert not splitter_directive.on_chip(offchip3)

    with pytest.raises(IndexError):
        splitter_directive.on_chip(nowhere)


def test_place_ensembles():
    # builder will move the learning stuff onto the host

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

    splitter_directive = SplitterDirective(net)
    assert not splitter_directive.on_chip(offchip)
    assert not splitter_directive.on_chip(direct)
    assert splitter_directive.on_chip(onchip)
    assert splitter_directive.on_chip(pre)
    assert splitter_directive.on_chip(post)
    assert splitter_directive.on_chip(error)

    with pytest.raises(TypeError):
        splitter_directive.on_chip(conn)


def test_place_internetwork_connections():
    with nengo.Network() as net:
        add_params(net)
        offchip = nengo.Ensemble(10, 1)
        net.config[offchip].on_chip = False
        onchip = nengo.Ensemble(10, 1)

        onon = nengo.Connection(onchip, onchip)
        onoff = nengo.Connection(onchip, offchip)
        offon = nengo.Connection(offchip, onchip)
        offoff = nengo.Connection(offchip, offchip)

    splitter_directive = SplitterDirective(net)

    assert splitter_directive.on_chip(onon.pre)
    assert splitter_directive.on_chip(onon.post)

    assert splitter_directive.on_chip(onoff.pre)
    assert not splitter_directive.on_chip(onoff.post)

    assert not splitter_directive.on_chip(offon.pre)
    assert splitter_directive.on_chip(offon.post)

    assert not splitter_directive.on_chip(offoff.pre)
    assert not splitter_directive.on_chip(offoff.post)


def test_split_host_to_learning_rule():
    with nengo.Network() as net:
        add_params(net)
        pre = nengo.Ensemble(10, 1, label="pre")
        post = nengo.Ensemble(10, 1, label="post")
        err_onchip = nengo.Ensemble(10, 1, label="err_onchip")
        err_offchip = nengo.Ensemble(10, 1, label="err_offchip")
        net.config[err_offchip].on_chip = False
        ens_conn = nengo.Connection(pre, post, learning_rule_type=nengo.PES())
        neurons_conn = nengo.Connection(pre.neurons, post.neurons,
                                        learning_rule_type=nengo.PES())
        nengo.Connection(err_onchip, ens_conn.learning_rule)
        nengo.Connection(
            err_onchip, neurons_conn.learning_rule)
        nengo.Connection(err_offchip, ens_conn.learning_rule)
        nengo.Connection(
            err_offchip, neurons_conn.learning_rule)

    splitter_directive = SplitterDirective(net)
    # builder will move these to host
    splitter_directive.move_to_host(err_onchip)
    splitter_directive.move_to_host(post)

    assert splitter_directive.on_chip(pre)
    assert not splitter_directive.on_chip(post)

    assert not splitter_directive.on_chip(err_onchip)
    assert not splitter_directive.on_chip(err_offchip)


def test_place_probes():
    with nengo.Network() as net:
        add_params(net)
        offchip1 = nengo.Node(0)
        with nengo.Network():
            onchip1 = nengo.Ensemble(10, 1)
            offchip2 = nengo.Ensemble(10, 1)
            net.config[offchip2].on_chip = False
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

    splitter_directive = SplitterDirective(net)
    assert all(not splitter_directive.on_chip(p) for p in offchip_probes)
    assert all(splitter_directive.on_chip(p) for p in onchip_probes)


def test_split_pre_from_host():
    with nengo.Network() as net:
        add_params(net)
        pre_1 = nengo.Node(0, label="pre_1")
        pre_2 = nengo.Ensemble(10, 1, label="pre_2")
        pre_3 = nengo.Node(size_in=1, label="pre_3")
        pre_4 = nengo.Ensemble(1, 1, label="pre_4")
        onchip = nengo.Ensemble(1, 1, label="onchip")
        post1 = nengo.Ensemble(10, 1, label="post1")
        post2 = nengo.Node(size_in=1, label="post2")

        nengo.Connection(pre_1, pre_2),
        nengo.Connection(pre_2, pre_3),
        nengo.Connection(pre_3, pre_4),
        nengo.Connection(pre_4.neurons, onchip),
        nengo.Connection(onchip, post1),
        nengo.Connection(post1, post2),

        net.config[pre_2].on_chip = False
        net.config[pre_4].on_chip = False

    splitter_directive = SplitterDirective(net, precompute=True)

    for obj in [pre_1, pre_2, pre_3, pre_4]:
        assert not splitter_directive.on_chip(obj)
        assert splitter_directive.is_precomputable(obj)

    for obj in [post1, post2]:
        assert not splitter_directive.on_chip(obj)
        assert not splitter_directive.is_precomputable(obj)

    assert splitter_directive.on_chip(onchip)
    with pytest.raises(ValueError):
        splitter_directive.is_precomputable(onchip)


def test_split_precompute_loop_error():
    with nengo.Network() as net:
        add_params(net)
        node_offchip = nengo.Node(lambda t, x: x + 1, size_in=1, size_out=1)
        ens_onchip = nengo.Ensemble(10, 1)
        nengo.Connection(node_offchip, ens_onchip)
        nengo.Connection(ens_onchip, node_offchip)

    with pytest.raises(BuildError, match="precompute"):
        SplitterDirective(net, precompute=True)


def test_already_moved_to_host():
    with nengo.Network() as net:
        u = nengo.Node(0)

    splitter_directive = SplitterDirective(net)
    with pytest.raises(ValueError):
        splitter_directive.move_to_host(u)
