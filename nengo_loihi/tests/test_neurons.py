import numpy as np
import nengo
import pytest

from nengo_loihi.neurons import (
    loihi_rates, LoihiLIF, LoihiSpikingRectifiedLinear)


@pytest.mark.parametrize('dt', [3e-4, 1e-3])
@pytest.mark.parametrize('neuron_type', [
    nengo.LIF(),
    nengo.LIF(tau_ref=0.001, tau_rc=0.07, amplitude=0.34),
    nengo.SpikingRectifiedLinear(),
    nengo.SpikingRectifiedLinear(amplitude=0.23),
])
def test_loihi_rates(dt, neuron_type, Simulator, plt, allclose):
    n = 256
    x = np.linspace(-0.1, 1, n)

    encoders = np.ones((n, 1))
    max_rates = 400 * np.ones(n)
    intercepts = 0 * np.ones(n)
    gain, bias = neuron_type.gain_bias(max_rates, intercepts)
    j = x * gain + bias

    with nengo.Network() as model:
        a = nengo.Ensemble(n, 1,
                           neuron_type=neuron_type,
                           encoders=encoders,
                           gain=gain,
                           bias=j)
        ap = nengo.Probe(a.neurons)

    with Simulator(model, dt=dt) as sim:
        sim.run(1.0)

    est_rates = sim.data[ap].mean(axis=0)
    ref_rates = loihi_rates(neuron_type, x, gain, bias, dt=dt)

    plt.plot(x, ref_rates, "k", label="predicted")
    plt.plot(x, est_rates, "g", label="measured")
    plt.legend(loc='best')

    assert allclose(est_rates, ref_rates, atol=1, rtol=0, xtol=1)


@pytest.mark.parametrize('neuron_type', [
    LoihiLIF(),
    LoihiSpikingRectifiedLinear(),
])
def test_loihi_neurons(neuron_type, Simulator, plt, allclose):
    dt = 0.0007

    n = 256
    encoders = np.ones((n, 1))
    gain = np.zeros(n)
    if isinstance(neuron_type, nengo.SpikingRectifiedLinear):
        bias = np.linspace(0, 1001, n)
    else:
        bias = np.linspace(0, 30, n)

    with nengo.Network() as model:
        a = nengo.Ensemble(n, 1, neuron_type=neuron_type,
                           encoders=encoders, gain=gain, bias=bias)
        ap = nengo.Probe(a.neurons)

    t_final = 1.0
    with nengo.Simulator(model, dt=dt) as nengo_sim:
        nengo_sim.run(t_final)

    with Simulator(model, dt=dt) as loihi_sim:
        loihi_sim.run(t_final)

    nengo_rates = (nengo_sim.data[ap] > 0).sum(axis=0) / t_final
    loihi_rates = (loihi_sim.data[ap] > 0).sum(axis=0) / t_final

    ref = neuron_type.rates(0., gain, bias, dt=dt)
    plt.plot(bias, loihi_rates, 'r', label='loihi sim')
    plt.plot(bias, nengo_rates, 'b-.', label='nengo sim')
    plt.plot(bias, ref, 'k--', label='ref')
    plt.legend(loc='best')

    atol = 1. / t_final  # the fundamental unit for our rates
    assert allclose(nengo_rates, ref, atol=atol, rtol=0, xtol=1)
    assert allclose(loihi_rates, ref, atol=atol, rtol=0, xtol=1)


@pytest.mark.parametrize(
    'neuron_type', [LoihiLIF(amplitude=1.0, tau_rc=0.02, tau_ref=0.002),
                    LoihiLIF(amplitude=0.063, tau_rc=0.05, tau_ref=0.001),
                    LoihiSpikingRectifiedLinear(),
                    LoihiSpikingRectifiedLinear(amplitude=0.42)])
def test_nengo_dl_neurons(neuron_type, plt, allclose):
    nengo_dl = pytest.importorskip('nengo_dl')
    from nengo_extras.neurons import SoftLIFRate
    import tensorflow as tf
    from tensorflow.python.ops import gradient_checker

    dt = 0.001
    nx = 256

    gain = 1
    bias = 0

    params = dict(amplitude=neuron_type.amplitude)
    if isinstance(neuron_type, LoihiLIF):
        x = np.linspace(-1, 30, nx)

        sigma = 0.02
        params.update(dict(tau_rc=neuron_type.tau_rc,
                           tau_ref=neuron_type.tau_ref))

        params2 = dict(params)
        params2['tau_ref'] = params2['tau_ref'] + 0.5*dt
    elif isinstance(neuron_type, LoihiSpikingRectifiedLinear):
        x = np.linspace(-1, 999, nx)

        tau_ref1 = 0.5*dt
        j = neuron_type.current(x, gain, bias) - 1

    with nengo.Network() as model:
        if isinstance(neuron_type, LoihiLIF):
            nengo_dl.configure_settings(lif_smoothing=sigma)

        u = nengo.Node([0] * nx)
        a = nengo.Ensemble(nx, 1, neuron_type=neuron_type,
                           gain=nengo.dists.Choice([gain]),
                           bias=nengo.dists.Choice([bias]))
        nengo.Connection(u, a.neurons, synapse=None)
        ap = nengo.Probe(a.neurons)

    # --- compute rates
    y_ref = loihi_rates(neuron_type, x, gain, bias, dt=dt)

    # y_med is an approximation of the smoothed Loihi tuning curve
    if isinstance(neuron_type, LoihiLIF):
        y_med = nengo.LIF(**params2).rates(x, gain, bias)
    elif isinstance(neuron_type, LoihiSpikingRectifiedLinear):
        y_med = np.zeros_like(j)
        y_med[j > 0] = neuron_type.amplitude / (tau_ref1 + 1./j[j > 0])

    with nengo_dl.Simulator(model, dt=dt) as sim:
        sim.run_steps(1, input_feeds={u: x[None, None, :]},
                      extra_feeds={sim.tensor_graph.signals.training: True})
        y = sim.data[ap][0]

    # --- compute spiking rates
    n_spike_steps = 1000
    x_spikes = x + np.zeros((1, n_spike_steps, 1), dtype=x.dtype)
    with nengo_dl.Simulator(model, dt=dt) as sim:
        sim.run_steps(n_spike_steps,
                      input_feeds={u: x_spikes},
                      extra_feeds={sim.tensor_graph.signals.training: False})
        y_spikes = sim.data[ap]
        y_spikerate = y_spikes.mean(axis=0)

    # --- compute derivates
    if isinstance(neuron_type, LoihiLIF):
        dy_ref = SoftLIFRate(sigma=sigma, **params2).derivative(x, gain, bias)
    else:
        # use the derivative of y_med (the smoothed Loihi tuning curve)
        dy_ref = np.zeros_like(j)
        dy_ref[j > 0] = neuron_type.amplitude / (j[j > 0]*tau_ref1 + 1)**2

    with nengo_dl.Simulator(model, dt=dt) as sim:
        n_steps = sim.unroll
        assert n_steps == 1

        inp = sim.tensor_graph.input_ph[u]
        inp_shape = inp.get_shape().as_list()
        inp_shape = [n_steps if s is None else s for s in inp_shape]
        inp_data = np.zeros(inp_shape) + x[None, :, None]

        out = sim.tensor_graph.probe_arrays[ap] + 0
        out_shape = out.get_shape().as_list()
        out_shape = [n_steps if s is None else s for s in out_shape]

        data = {n: np.zeros((sim.minibatch_size, n_steps, n.size_out))
                for n in sim.tensor_graph.invariant_inputs}
        data.update({p: np.zeros((sim.minibatch_size, n_steps, p.size_in))
                     for p in sim.tensor_graph.target_phs})
        feed = sim._fill_feed(n_steps, data=data, training=True)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            dx, dy = gradient_checker._compute_dx_and_dy(inp, out, out_shape)
            sim.sess.run(tf.variables_initializer(
                scope.get_collection("gradient_vars")))

        with sim.sess.as_default():
            analytic = gradient_checker._compute_theoretical_jacobian(
                inp, inp_shape, inp_data, dy, out_shape, dx,
                extra_feed_dict=feed)

        dy = np.array(np.diag(analytic))

    dx = x[1] - x[0]
    dy_est = np.diff(nengo.synapses.Alpha(10).filtfilt(y_ref, dt=1)) / dx
    x1 = 0.5*(x[:-1] + x[1:])

    # --- plots
    plt.subplot(211)
    plt.plot(x, y_med, '--', label='LIF(tau_ref += 0.5*dt)')
    plt.plot(x, y, label='nengo_dl')
    plt.plot(x, y_spikerate, label='nengo_dl spikes')
    plt.plot(x, y_ref, 'k--', label='LoihiLIF')
    plt.legend(loc=4)

    plt.subplot(212)
    plt.plot(x1, dy_est, '--', label='diff(smoothed_y)')
    plt.plot(x, dy, label='nengo_dl')
    plt.plot(x, dy_ref, 'k--', label='diff(SoftLIF)')
    plt.legend(loc=1)

    np.fill_diagonal(analytic, 0)
    assert np.all(analytic == 0)

    assert allclose(y, y_ref, atol=1e-3, rtol=1e-5)
    assert allclose(dy, dy_ref, atol=1e-3, rtol=1e-5)
    assert allclose(y_spikerate, y_ref, atol=1, rtol=1e-2)


@pytest.mark.parametrize(
    'neuron_type', [
        LoihiLIF(amplitude=0.3, nengo_dl_noise_model=('alpha_rc', 0.001)),
    ])
def test_nengo_dl_noise(neuron_type, plt, allclose):
    nengo_dl = pytest.importorskip('nengo_dl')
    from nengo_extras.neurons import SoftLIFRate
    import tensorflow as tf

    dt = 0.001
    nx = 256

    gain = 1
    bias = 0

    x = np.linspace(-1, 30, nx)

    params = dict(amplitude=neuron_type.amplitude)
    params.update(dict(tau_rc=neuron_type.tau_rc,
                       tau_ref=neuron_type.tau_ref))
    params2 = dict(params)
    params2['tau_ref'] = params2['tau_ref'] + 0.5*dt

    sigma = 0.02

    with nengo.Network() as model:
        u = nengo.Node([0] * nx)
        a = nengo.Ensemble(nx, 1, neuron_type=neuron_type,
                           gain=nengo.dists.Choice([gain]),
                           bias=nengo.dists.Choice([bias]))
        nengo.Connection(u, a.neurons, synapse=None)
        ap = nengo.Probe(a.neurons)

    # --- compute rates
    y_ref = loihi_rates(neuron_type, x, gain, bias, dt=dt)
    y_med = nengo.LIF(**params2).rates(x, gain, bias)

    with nengo_dl.Simulator(model, dt=dt) as sim:
        input_data = {u: np.tile(x[None, None, :], (500, 1, 1))}
        outputs = {ap: None}
        y = sim.run_batch(input_data, outputs, training=True)[ap]
        y = y[:, 0, 0, :]

    ymean = y.mean(axis=0)
    y25 = np.percentile(y, 25, axis=0)
    y75 = np.percentile(y, 75, axis=0)

    # --- plots
    plt.plot(x, y_med, '--', label='LIF(tau_ref += 0.5*dt)')
    plt.plot(x, ymean, label='nengo_dl')
    plt.plot(x, y25, ':', label='25th')
    plt.plot(x, y75, ':', label='75th')
    plt.plot(x, y_ref, 'k--', label='LoihiLIF')
    plt.legend()
