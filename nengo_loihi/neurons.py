import logging

from nengo.neurons import LIF, SpikingRectifiedLinear
import numpy as np

logger = logging.getLogger(__name__)

try:
    import nengo_dl
    import nengo_dl.neuron_builders
    import tensorflow as tf
except ImportError as e:  # pragma: no cover
    logger.debug("Error import nengo_dl/tensorflow: %s", e)
    nengo_dl = None
    tf = None


def discretize_tau_rc(tau_rc, dt):
    """Discretize tau_rc as per discretize_compartment.

    Parameters
    ----------
    tau_rc : float
        The neuron membrane time constant.
    dt : float
        The simulator time step.
    """
    lib = tf if tf is not None and isinstance(tau_rc, tf.Tensor) else np

    decay_rc = -lib.expm1(-dt / tau_rc)
    decay_rc = lib.round(decay_rc * (2**12 - 1)) / (2**12 - 1)
    return -dt / lib.log1p(-decay_rc)


def discretize_tau_ref(tau_ref, dt):
    """Discretize tau_ref as per Compartment.configure_lif.

    Parameters
    ----------
    tau_rc : float
        The neuron membrane time constant.
    dt : float
        The simulator time step.
    """
    lib = tf if tf is not None and isinstance(tau_ref, tf.Tensor) else np

    return dt * lib.round(tau_ref / dt)


def loihi_lif_rates(neuron_type, x, gain, bias, dt):
    tau_ref = discretize_tau_ref(neuron_type.tau_ref, dt)
    tau_rc = discretize_tau_rc(neuron_type.tau_rc, dt)

    j = neuron_type.current(x, gain, bias) - 1
    out = np.zeros_like(j)
    period = tau_ref + tau_rc * np.log1p(1. / j[j > 0])
    out[j > 0] = (neuron_type.amplitude / dt) / np.ceil(period / dt)
    return out


def loihi_spikingrectifiedlinear_rates(neuron_type, x, gain, bias, dt):
    j = neuron_type.current(x, gain, bias)

    out = np.zeros_like(j)
    period = 1. / j[j > 0]
    out[j > 0] = (neuron_type.amplitude / dt) / np.ceil(period / dt)
    return out


def loihi_rates(neuron_type, x, gain, bias, dt):
    for cls in type(neuron_type).__mro__:
        if cls in loihi_rate_functions:
            return loihi_rate_functions[cls](neuron_type, x, gain, bias, dt)
    return neuron_type.rates(x, gain, bias)


loihi_rate_functions = {
    LIF: loihi_lif_rates,
    SpikingRectifiedLinear: loihi_spikingrectifiedlinear_rates,
}


class LoihiLIF(LIF):
    """Simulate LIF neurons as done by Loihi.

    On Loihi, the inter-spike interval has to be an integer. This causes
    aliasing the firing rates where a wide variety of inputs can produce the
    same output firing rate. This class reproduces this effect, as well as
    the discretization of some of the neuron parameters. It can be used in
    e.g. ``nengo`` or ``nengo_dl`` to reproduce these unique Loihi effects.

    Parameters
    ----------
    nengo_dl_noise : NeuronOutputNoise
        Noise added to the rate-neuron output when training with this neuron
        type in ``nengo_dl``.
    """

    def __init__(self, tau_rc=0.02, tau_ref=0.002, min_voltage=0, amplitude=1,
                 nengo_dl_noise=None):
        super(LoihiLIF, self).__init__(
            tau_rc=tau_rc, tau_ref=tau_ref, min_voltage=min_voltage,
            amplitude=amplitude)
        self.nengo_dl_noise = nengo_dl_noise

    @property
    def _argreprs(self):
        args = super(LoihiLIF, self)._argreprs
        if self.nengo_dl_noise is not None:
            args.append("nengo_dl_noise=%s" % self.nengo_dl_noise)
        return args

    def rates(self, x, gain, bias, dt=0.001):
        return loihi_lif_rates(self, x, gain, bias, dt)

    def step_math(self, dt, J, spiked, voltage, refractory_time):
        tau_ref = discretize_tau_ref(self.tau_ref, dt)

        refractory_time -= dt
        delta_t = (dt - refractory_time).clip(0, dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / self.tau_rc)

        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        voltage[voltage < self.min_voltage] = self.min_voltage
        voltage[spiked_mask] = 0
        refractory_time[spiked_mask] = tau_ref + dt


class LoihiSpikingRectifiedLinear(SpikingRectifiedLinear):
    """Simulate spiking Rectified Linear neurons as done by Loihi.

    On Loihi, the inter-spike interval has to be an integer. This causes
    aliasing the firing rates where a wide variety of inputs can produce the
    same output firing rate. This class reproduces this effect. It can be used
    in e.g. ``nengo`` or ``nengo_dl`` to reproduce these unique Loihi effects.
    """

    def rates(self, x, gain, bias, dt=0.001):
        return loihi_spikingrectifiedlinear_rates(self, x, gain, bias, dt)

    def step_math(self, dt, J, spiked, voltage):
        voltage += J * dt

        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        voltage[voltage < 0] = 0
        voltage[spiked_mask] = 0


class NeuronOutputNoise(object):
    """Noise added to the output of a rate neuron.

    Often used when training deep networks with rate neurons for final
    implementation in spiking neurons, to simulate the variability
    caused by the spiking neurons.
    """
    pass


class LowpassRCNoise(NeuronOutputNoise):
    """Noise model combining Lowpass synapse and neuron membrane filters.

    Samples "noise" (i.e. variability) from a regular spike train filtered
    by the following transfer function, where :math:`\tau_{rc}` is the
    membrane time constant and :math:`\tau_s` is the synapse time constant:

    .. math::

        H(s) = [(\tau_s s + 1) (\tau_{rc} s + 1)]^{-1}

    See [1]_ for background and derivations.

    Attributes
    ----------
    tau_s : float
        Time constant for Lowpass synaptic filter.

    References
    ----------
    .. [1] E. Hunsberger (2018) "Spiking Deep Neural Networks: Engineered and
       Biological Approaches to Object Recognition." PhD thesis. pp. 106--113.
       (http://compneuro.uwaterloo.ca/publications/hunsberger2018.html)
    """
    def __init__(self, tau_s):
        self.tau_s = tau_s

    def __repr__(self):
        return "%s(tau_s=%s)" % (type(self).__name__, self.tau_s)


class AlphaRCNoise(NeuronOutputNoise):
    """Noise model combining Alpha synapse and neuron membrane filters.

    Samples "noise" (i.e. variability) from a regular spike train filtered
    by the following transfer function, where :math:`\tau_{rc}` is the
    membrane time constant and :math:`\tau_s` is the synapse time constant:

    .. math::

        H(s) = [(\tau_s s + 1)^2 (\tau_rc s + 1)]^{-1}

    See [1]_ for background and derivations.

    Attributes
    ----------
    tau_s : float
        Time constant for Alpha synaptic filter.

    References
    ----------
    .. [1] E. Hunsberger (2018) "Spiking Deep Neural Networks: Engineered and
       Biological Approaches to Object Recognition." PhD thesis. pp. 106--113.
       (http://compneuro.uwaterloo.ca/publications/hunsberger2018.html)
    """
    def __init__(self, tau_s):
        self.tau_s = tau_s

    def __repr__(self):
        return "%s(tau_s=%s)" % (type(self).__name__, self.tau_s)


if nengo_dl is not None:  # noqa: C901
    class NoiseBuilder(object):
        """Build noise classes in ``nengo_dl``.

        Attributes
        ----------
        noise_models : list of NeuronOutputNoise
            The noise models used for each op/signal.
        """

        builders = {}

        def __init__(self, ops, signals, config, noise_models):
            self.noise_models = noise_models
            self.dtype = signals.dtype
            self.np_dtype = self.dtype.as_numpy_dtype()

        @classmethod
        def build(cls, ops, signals, config):
            """Create a NoiseBuilder for the provided ops."""

            noise_models = [getattr(op.neurons, 'nengo_dl_noise', None)
                            for op in ops]
            model_type = (type(noise_models[0]) if len(noise_models) > 0
                          else None)
            equal_types = all(type(m) is model_type for m in noise_models)

            if not equal_types:
                raise NotImplementedError(
                    "Multiple noise models for the same neuron type is "
                    "not supported")

            return cls.builders[model_type](ops, signals, config, noise_models)

        def generate(self, period, tau_rc=None):
            """Generate TensorFlow code to implement these noise models.

            Parameters
            ----------
            period : tf.Tensor
                The inter-spike periods of the neurons to add noise to.
            tau_rc : tf.Tensor
                The membrane time constant of the neurons (used by some noise
                models).
            """
            raise NotImplementedError("Subclass must implement")

    class NoNoiseBuilder(NoiseBuilder):
        """nengo_dl builder for if there is no noise model."""

        def generate(self, period, tau_rc=None):
            return tf.reciprocal(period)

    class LowpassRCNoiseBuilder(NoiseBuilder):
        """nengo_dl builder for the LowpassRCNoise model."""

        def __init__(self, ops, signals, *args, **kwargs):
            super(LowpassRCNoiseBuilder, self).__init__(
                ops, signals, *args, **kwargs)

            # tau_s is the time constant of the synaptic filter
            tau_s = np.concatenate([
                model.tau_s * np.ones((op.J.shape[0], 1),
                                      dtype=self.np_dtype)
                for model, op in zip(self.noise_models, ops)])
            self.tau_s = signals.constant(tau_s, dtype=self.dtype)

        def generate(self, period, tau_rc=None):
            d = tau_rc - self.tau_s
            u01 = tf.random_uniform(tf.shape(period))
            t = u01 * period
            q_rc = tf.exp(-t / tau_rc)
            q_s = tf.exp(-t / self.tau_s)
            r_rc1 = -tf.expm1(-period / tau_rc)  # 1 - exp(-period/tau_rc)
            r_s1 = -tf.expm1(-period / self.tau_s)  # 1 - exp(-period/tau_s)
            return (1./d) * (q_rc/r_rc1 - q_s/r_s1)

    class AlphaRCNoiseBuilder(NoiseBuilder):
        """nengo_dl builder for the AlphaRCNoise model."""

        def __init__(self, ops, signals, *args, **kwargs):
            super(AlphaRCNoiseBuilder, self).__init__(
                ops, signals, *args, **kwargs)

            # tau_s is the time constant of the synaptic filter
            tau_s = np.concatenate([
                model.tau_s * np.ones((op.J.shape[0], 1),
                                      dtype=self.np_dtype)
                for model, op in zip(self.noise_models, ops)])
            self.tau_s = signals.constant(tau_s, dtype=self.dtype)

        def generate(self, period, tau_rc=None):
            d = tau_rc - self.tau_s
            u01 = tf.random_uniform(tf.shape(period))
            t = u01 * period
            q_rc = tf.exp(-t / tau_rc)
            q_s = tf.exp(-t / self.tau_s)
            r_rc1 = -tf.expm1(-period / tau_rc)  # 1 - exp(-period/tau_rc)
            r_s1 = -tf.expm1(-period / self.tau_s)  # 1 - exp(-period/tau_s)

            pt = tf.where(period < 100*self.tau_s, (period - t)*(1 - r_s1),
                          tf.zeros_like(period))
            qt = tf.where(t < 100*self.tau_s, q_s*(t + pt), tf.zeros_like(t))
            rt = qt / (self.tau_s * d * r_s1**2)
            rn = tau_rc*(q_rc/(d*d*r_rc1) - q_s/(d*d*r_s1)) - rt
            return rn

    NoiseBuilder.builders[type(None)] = NoNoiseBuilder
    NoiseBuilder.builders[LowpassRCNoise] = LowpassRCNoiseBuilder
    NoiseBuilder.builders[AlphaRCNoise] = AlphaRCNoiseBuilder

    class LoihiLIFBuilder(nengo_dl.neuron_builders.LIFBuilder):
        """nengo_dl builder for the LoihiLIF neuron type.

        Attributes
        ----------
        spike_noise : NoiseBuilder
            Generator for any output noise associated with these neurons.
        """
        def __init__(self, ops, signals, config):
            super(LoihiLIFBuilder, self).__init__(ops, signals, config)

            self.spike_noise = NoiseBuilder.build(ops, signals, config)

        def _rate_step(self, J, dt):
            tau_ref = discretize_tau_ref(self.tau_ref, dt)
            tau_rc = discretize_tau_rc(self.tau_rc, dt)

            # Since LoihiLIF takes `ceil(period/dt)` the firing rate is
            # always below the LIF rate. Using `tau_ref1` in LIF curve makes
            # it the average of the LoihiLIF curve (rather than upper bound).
            tau_ref1 = tau_ref + 0.5*dt

            J -= self.one

            # --- compute Loihi rates (for forward pass)
            period = tau_ref + tau_rc*tf.log1p(tf.reciprocal(
                tf.maximum(J, self.epsilon)))
            period = dt * tf.ceil(period / dt)
            loihi_rates = self.spike_noise.generate(period, tau_rc=tau_rc)
            loihi_rates = tf.where(J > self.zero, self.amplitude * loihi_rates,
                                   self.zeros)

            # --- compute LIF rates (for backward pass)
            if self.config.lif_smoothing:
                js = J / self.sigma
                j_valid = js > -20
                js_safe = tf.where(j_valid, js, self.zeros)

                # softplus(js) = log(1 + e^js)
                z = tf.nn.softplus(js_safe) * self.sigma

                # as z->0
                #   z = s*log(1 + e^js) = s*e^js
                #   log(1 + 1/z) = log(1/z) = -log(s*e^js) = -js - log(s)
                q = tf.where(j_valid,
                             tf.log1p(tf.reciprocal(z)),
                             -js - tf.log(self.sigma))

                rates = self.amplitude / (tau_ref1 + tau_rc*q)
            else:
                rates = self.amplitude / (
                    tau_ref1 + tau_rc*tf.log1p(tf.reciprocal(
                        tf.maximum(J, self.epsilon))))
                rates = tf.where(J > self.zero, rates, self.zeros)

            # rates + stop_gradient(loihi_rates - rates) =
            #     loihi_rates on forward pass, rates on backwards
            return rates + tf.stop_gradient(loihi_rates - rates)

        def _step(self, J, voltage, refractory, dt):
            tau_ref = discretize_tau_ref(self.tau_ref, dt)
            tau_rc = discretize_tau_rc(self.tau_rc, dt)

            delta_t = tf.clip_by_value(dt - refractory, self.zero, dt)
            voltage -= (J - voltage) * tf.expm1(-delta_t / tau_rc)

            spiked = voltage > self.one
            spikes = tf.cast(spiked, J.dtype) * self.alpha

            refractory = tf.where(spiked, tau_ref + self.zeros,
                                  refractory - dt)
            voltage = tf.where(spiked, self.zeros,
                               tf.maximum(voltage, self.min_voltage))

            # we use stop_gradient to avoid propagating any nans (those get
            # propagated through the cond even if the spiking version isn't
            # being used at all)
            return (tf.stop_gradient(spikes), tf.stop_gradient(voltage),
                    tf.stop_gradient(refractory))

        def build_step(self, signals):
            J = signals.gather(self.J_data)
            voltage = signals.gather(self.voltage_data)
            refractory = signals.gather(self.refractory_data)

            spike_out, spike_voltage, spike_ref = self._step(
                J, voltage, refractory, signals.dt)

            if self.config.inference_only:
                spikes, voltage, refractory = (
                    spike_out, spike_voltage, spike_ref)
            else:
                rate_out = self._rate_step(J, signals.dt)

                spikes, voltage, refractory = tf.cond(
                    signals.training,
                    lambda: (rate_out, voltage, refractory),
                    lambda: (spike_out, spike_voltage, spike_ref)
                )

            signals.scatter(self.output_data, spikes)
            signals.mark_gather(self.J_data)
            signals.scatter(self.refractory_data, refractory)
            signals.scatter(self.voltage_data, voltage)

    class LoihiSpikingRectifiedLinearBuilder(
            nengo_dl.neuron_builders.SpikingRectifiedLinearBuilder):
        """nengo_dl builder for the LoihiSpikingRectifiedLinear neuron type.
        """

        def __init__(self, ops, signals, config):
            super(LoihiSpikingRectifiedLinearBuilder, self).__init__(
                ops, signals, config)

            self.amplitude = signals.op_constant(
                [op.neurons for op in ops], [op.J.shape[0] for op in ops],
                "amplitude", signals.dtype)

            self.zeros = tf.zeros(
                self.J_data.shape + (signals.minibatch_size,),
                signals.dtype)

            self.epsilon = tf.constant(1e-15, dtype=signals.dtype)

            # copy these so that they're easily accessible in _step functions
            self.zero = signals.zero
            self.one = signals.one

        def _rate_step(self, J, dt):
            # Since LoihiLIF takes `ceil(period/dt)` the firing rate is
            # always below the LIF rate. Using `tau_ref1` in LIF curve makes
            # it the average of the LoihiLIF curve (rather than upper bound).
            tau_ref1 = 0.5*dt

            # --- compute Loihi rates (for forward pass)
            period = tf.reciprocal(tf.maximum(J, self.epsilon))
            loihi_rates = self.alpha / tf.ceil(period / dt)
            loihi_rates = tf.where(J > self.zero, loihi_rates, self.zeros)

            # --- compute RectifiedLinear rates (for backward pass)
            rates = self.amplitude / (
                tau_ref1 + tf.reciprocal(tf.maximum(J, self.epsilon)))
            rates = tf.where(J > self.zero, rates, self.zeros)

            # rates + stop_gradient(loihi_rates - rates) =
            #     loihi_rates on forward pass, rates on backwards
            return rates + tf.stop_gradient(loihi_rates - rates)

        def _step(self, J, voltage, dt):
            voltage += J * dt
            spiked = voltage > self.one
            spikes = tf.cast(spiked, J.dtype) * self.alpha
            voltage = tf.where(spiked, self.zeros, tf.maximum(voltage, 0))

            # we use stop_gradient to avoid propagating any nans (those get
            # propagated through the cond even if the spiking version isn't
            # being used at all)
            return tf.stop_gradient(spikes), tf.stop_gradient(voltage)

        def build_step(self, signals):
            J = signals.gather(self.J_data)
            voltage = signals.gather(self.voltage_data)

            spike_out, spike_voltage = self._step(J, voltage, signals.dt)

            if self.config.inference_only:
                out, voltage = spike_out, spike_voltage
            else:
                rate_out = self._rate_step(J, signals.dt)

                out, voltage = tf.cond(
                    signals.training,
                    lambda: (rate_out, voltage),
                    lambda: (spike_out, spike_voltage))

            signals.scatter(self.output_data, out)
            signals.scatter(self.voltage_data, voltage)

    nengo_dl.neuron_builders.SimNeuronsBuilder.TF_NEURON_IMPL[
        LoihiLIF] = LoihiLIFBuilder
    nengo_dl.neuron_builders.SimNeuronsBuilder.TF_NEURON_IMPL[
        LoihiSpikingRectifiedLinear] = LoihiSpikingRectifiedLinearBuilder
