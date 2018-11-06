import nengo
import numpy as np
import pytest


@pytest.mark.parametrize('n_per_dim', [120, 200])
@pytest.mark.parametrize('dims', [1, 3])
def test_pes_comm_channel(allclose, plt, seed, Simulator, n_per_dim, dims):
    scale = np.linspace(1, 0, dims + 1)[:-1]
    input_fn = lambda t: np.sin(t * 2 * np.pi) * scale

    tau = 0.01
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(input_fn)

        pre = nengo.Ensemble(n_per_dim * dims, dims)
        post = nengo.Node(size_in=dims)

        nengo.Connection(stim, pre, synapse=None)
        conn = nengo.Connection(
            pre, post,
            function=lambda x: np.zeros(dims),
            synapse=tau,
            learning_rule_type=nengo.PES(learning_rate=1e-3))

        nengo.Connection(post, conn.learning_rule)
        nengo.Connection(stim, conn.learning_rule, transform=-1)

        p_stim = nengo.Probe(stim, synapse=0.02)
        p_pre = nengo.Probe(pre, synapse=0.02)
        p_post = nengo.Probe(post, synapse=0.02)

    simtime = 5.0
    with nengo.Simulator(model) as nengo_sim:
        nengo_sim.run(simtime)

    with Simulator(model, precompute=False) as loihi_sim:
        loihi_sim.run(simtime)

    t = nengo_sim.trange()
    pre_tmask = t > 0.1
    post_tmask = t > simtime - 1.0

    inter_tau = loihi_sim.model.inter_tau
    y = nengo_sim.data[p_stim]
    y_dpre = nengo.Lowpass(inter_tau).filt(y)
    y_dpost = nengo.Lowpass(tau).combine(nengo.Lowpass(inter_tau)).filt(y_dpre)
    y_nengo = nengo_sim.data[p_post]
    y_loihi = loihi_sim.data[p_post]

    plt.subplot(211)
    plt.plot(t, y_dpost, 'k', label='target')
    plt.plot(t, y_nengo, 'b', label='nengo')
    plt.plot(t, y_loihi, 'g', label='loihi')

    plt.subplot(212)
    plt.plot(t[post_tmask], y_loihi[post_tmask] - y_dpost[post_tmask], 'k')
    plt.plot(t[post_tmask], y_loihi[post_tmask] - y_nengo[post_tmask], 'b')

    assert allclose(loihi_sim.data[p_pre][pre_tmask], y_dpre[pre_tmask],
                    atol=0.1, rtol=0.05)
    assert allclose(y_loihi[post_tmask], y_dpost[post_tmask],
                    atol=0.1, rtol=0.05)
    assert allclose(y_loihi, y_nengo, atol=0.15, rtol=0.1)


@pytest.mark.parametrize('init_function', [None, lambda x: 0])
def test_multiple_pes(init_function, allclose, plt, seed, Simulator):
    n_errors = 5
    targets = np.linspace(-1, 1, n_errors)
    with nengo.Network(seed=seed) as model:
        pre_ea = nengo.networks.EnsembleArray(200, n_ensembles=n_errors)
        errors = nengo.Node(size_in=n_errors)
        output = nengo.Node(size_in=n_errors)

        target = nengo.Node(targets)
        nengo.Connection(target, errors, transform=-1)
        nengo.Connection(output, errors)

        for i in range(n_errors):
            conn = nengo.Connection(
                pre_ea.ea_ensembles[i],
                output[i],
                function=init_function,
                learning_rule_type=nengo.PES(learning_rate=3e-3),
            )
            nengo.Connection(target[i], conn.learning_rule, transform=-1)
            nengo.Connection(output[i], conn.learning_rule)

        probe = nengo.Probe(output, synapse=0.1)

    simtime = 2.5
    with Simulator(model, precompute=False) as sim:
        sim.run(simtime)

    t = sim.trange()
    tmask = t > simtime * 0.85

    plt.plot(t, sim.data[probe])
    for target, style in zip(targets, plt.rcParams["axes.prop_cycle"]):
        plt.axhline(target, **style)

    for i, target in enumerate(targets):
        assert allclose(sim.data[probe][tmask, i], target,
                        atol=0.05, rtol=0.05), "Target %d not close" % i
