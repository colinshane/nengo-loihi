import warnings

import numpy as np

import nengo
from nengo.exceptions import ValidationError
from nengo.neurons import NeuronType
from nengo.params import NumberParam

try:
    import nengo_dl
    import nengo_dl.neuron_builders
    import tensorflow as tf
except ImportError:
    nengo_dl = None
    tf = None


def loihi_lif_rates(neuron_type, x, gain, bias, dt):
    # discretize tau_ref as per CxGroup.configure_lif
    tau_ref = dt * np.round(neuron_type.tau_ref / dt)

    # discretize tau_rc as per CxGroup.discretize
    decay_rc = -np.expm1(-dt/neuron_type.tau_rc)
    decay_rc = np.round(decay_rc * (2**12 - 1)) / (2**12 - 1)
    tau_rc = -dt/np.log1p(-decay_rc)

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
    nengo.LIF: loihi_lif_rates,
    nengo.SpikingRectifiedLinear: loihi_spikingrectifiedlinear_rates,
}


class LoihiLIF(nengo.LIF):
    def __init__(self, *args, nengo_dl_noise=None, **kwargs):
        super(LoihiLIF, self).__init__(*args, **kwargs)
        self.nengo_dl_noise = nengo_dl_noise

    def rates(self, x, gain, bias, dt=0.001):
        return loihi_lif_rates(self, x, gain, bias, dt)

    def step_math(self, dt, J, spiked, voltage, refractory_time):
        tau_ref = dt * np.round(self.tau_ref / dt)
        refractory_time -= dt

        delta_t = (dt - refractory_time).clip(0, dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / self.tau_rc)

        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        voltage[voltage < self.min_voltage] = self.min_voltage
        voltage[spiked_mask] = 0
        refractory_time[spiked_mask] = tau_ref + dt


class LoihiSpikingRectifiedLinear(nengo.SpikingRectifiedLinear):
    def rates(self, x, gain, bias, dt=0.001):
        return loihi_spikingrectifiedlinear_rates(self, x, gain, bias, dt)

    def step_math(self, dt, J, spiked, voltage):
        voltage += J * dt

        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        voltage[voltage < 0] = 0
        voltage[spiked_mask] = 0


class NIFRate(NeuronType):
    """Non-spiking version of the non-leaky integrate-and-fire (NIF) model.

    Parameters
    ----------
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    """

    probeable = ('rates',)

    tau_ref = NumberParam('tau_ref', low=0)
    amplitude = NumberParam('amplitude', low=0, low_open=True)

    def __init__(self, tau_ref=0.002, amplitude=1):
        super(NIFRate, self).__init__()
        self.tau_ref = tau_ref
        self.amplitude = amplitude

    @property
    def _argreprs(self):
        args = []
        if self.tau_ref != 0.002:
            args.append("tau_ref=%s" % self.tau_ref)
        if self.amplitude != 1:
            args.append("amplitude=%s" % self.amplitude)
        return args

    def gain_bias(self, max_rates, intercepts):
        """Analytically determine gain, bias."""
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)

        inv_tau_ref = 1. / self.tau_ref if self.tau_ref > 0 else np.inf
        if np.any(max_rates > inv_tau_ref):
            raise ValidationError("Max rates must be below the inverse "
                                  "refractory period (%0.3f)" % inv_tau_ref,
                                  attr='max_rates', obj=self)

        x = 1.0 / (1.0/max_rates - self.tau_ref)
        gain = x / (1 - intercepts)
        bias = 1 - gain * intercepts
        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        """Compute the inverse of gain_bias."""
        intercepts = (1 - bias) / gain
        max_rates = 1.0 / (self.tau_ref + 1.0/(gain + bias - 1))
        if not np.all(np.isfinite(max_rates)):
            warnings.warn("Non-finite values detected in `max_rates`; this "
                          "probably means that `gain` was too small.")
        return max_rates, intercepts

    def rates(self, x, gain, bias):
        """Always use NIFRate to determine rates."""
        J = self.current(x, gain, bias)
        out = np.zeros_like(J)
        # Use NIFRate's step_math explicitly to ensure rate approximation
        NIFRate.step_math(self, dt=1, J=J, output=out)
        return out

    def step_math(self, dt, J, output):
        """Implement the NIFRate nonlinearity."""
        j = J - 1
        output[:] = 0  # faster than output[j <= 0] = 0
        output[j > 0] = self.amplitude / (self.tau_ref + 1./j[j > 0])


# class NIF(NIFRate):
#     """Spiking version of non-leaky integrate-and-fire (NIF) neuron model.

#     Parameters
#     ----------
#     tau_ref : float
#         Absolute refractory period, in seconds. This is how long the
#         membrane voltage is held at zero after a spike.
#     min_voltage : float
#         Minimum value for the membrane voltage. If ``-np.inf``, the voltage
#         is never clipped.
#     amplitude : float
#         Scaling factor on the neuron output. Corresponds to the relative
#         amplitude of the output spikes of the neuron.
#     """

#     probeable = ('spikes', 'voltage', 'refractory_time')

#     min_voltage = NumberParam('min_voltage', high=0)

#     def __init__(self, tau_ref=0.002, min_voltage=0, amplitude=1):
#         super(NIF, self).__init__(tau_ref=tau_ref, amplitude=amplitude)
#         self.min_voltage = min_voltage

#     def step_math(self, dt, J, spiked, voltage, refractory_time):
#         refractory_time -= dt
#         delta_t = (dt - refractory_time).clip(0, dt)
#         voltage += J * delta_t

#         # determine which neurons spiked (set them to 1/dt, else 0)
#         spiked_mask = voltage > 1
#         spiked[:] = spiked_mask * (self.amplitude / dt)

#         # set v(0) = 1 and solve for t to compute the spike time
#         t_spike = dt - (voltage[spiked_mask] - 1) / J[spiked_mask]

#         # set spiked voltages to zero, refractory times to tau_ref, and
#         # rectify negative voltages to a floor of min_voltage
#         voltage[voltage < self.min_voltage] = self.min_voltage
#         voltage[spiked_mask] = 0
#         refractory_time[spiked_mask] = self.tau_ref + t_spike


class NIF(NIFRate):
    probeable = ('spikes', 'voltage', 'refractory_time')

    min_voltage = NumberParam('min_voltage', high=0)

    def __init__(self, tau_ref=0.002, min_voltage=0, amplitude=1):
        super(NIF, self).__init__(tau_ref=tau_ref, amplitude=amplitude)
        self.min_voltage = min_voltage

    def step_math(self, dt, J, spiked, voltage, refractory_time):
        refractory_time -= dt
        delta_t = (dt - refractory_time).clip(0, dt)
        voltage += J * delta_t

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        voltage[voltage < self.min_voltage] = self.min_voltage
        voltage[spiked_mask] -= 1
        refractory_time[spiked_mask] = self.tau_ref + dt


@nengo.builder.Builder.register(NIFRate)
def nengo_build_nif_rate(model, nif_rate, neurons):
    return nengo.builder.neurons.build_lif(model, nif_rate, neurons)


@nengo.builder.Builder.register(NIF)
def nengo_build_nif(model, nif, neurons):
    return nengo.builder.neurons.build_lif(model, nif, neurons)


if nengo_dl is not None:  # noqa: C901
    class LowpassRCNoise(object):
        """Noise model combining Lowpass synapse and neuron membrane

        Parameters
        ----------
        tau_s : float
            Time constant for Lowpass synaptic filter.
        """
        def __init__(self, tau_s):
            self.tau_s = tau_s


    class AlphaRCNoise(object):
        """Noise model combining Alpha synapse and neuron membrane

        Parameters
        ----------
        tau_s : float
            Time constant for Alpha synaptic filter.
        """
        def __init__(self, tau_s):
            self.tau_s = tau_s


    class NoiseBuilder(object):
        builders = {}

        def __init__(self, models, ops, signals, config):
            self.models = models
            self.dtype = signals.dtype
            self.np_dtype = self.dtype.as_numpy_dtype()
            self.np_ones = [
                np.ones((op.J.shape[0], 1), dtype=self.np_dtype) for op in ops]

        @classmethod
        def build(cls, ops, signals, config):
            models = [getattr(op.neurons, 'nengo_dl_noise', None)
                      for op in ops]
            model_type = type(models[0]) if len(models) > 0 else None
            equal_types = all(type(m) is model_type for m in models)

            if equal_types and model_type in cls.builders:
                return cls.builders[model_type](models, ops, signals, config)
            else:
                return NoiseBuilder(models, ops, signals, config)

        def generate(self, period, tau_rc=None):
            raise NotImplementedError(
                "Multiple noise models not supported for the same neuron "
                "type")


    class NoNoiseBuilder(NoiseBuilder):
        def generate(self, period, tau_rc=None):
            return tf.reciprocal(period)


    class RCNoiseBuilder(NoiseBuilder):
        def __init__(self, models, ops, signals, config):
            super(RCNoiseBuilder, self).__init__(
                models, ops, signals, config)

            tau_s = np.concatenate([
                model.tau_s * one
                for model, one in zip(self.models, self.np_ones)])
            self.tau_s = signals.constant(tau_s, dtype=self.dtype)

        @classmethod
        def tensorflow(cls, period, tau_s, tau_rc):
            raise NotImplementedError("Subclass must implement")

        def generate(self, period, tau_rc=None):
            assert tau_rc is not None
            return self.tensorflow(period, self.tau_s, tau_rc)


    class LowpassRCNoiseBuilder(RCNoiseBuilder):
        @classmethod
        def tensorflow(cls, period, tau_s, tau_rc):
            d = tau_rc - tau_s
            u01 = tf.random_uniform(tf.shape(period))
            t = u01 * period
            q_rc = tf.exp(-t / tau_rc)
            q_s = tf.exp(-t / tau_s)
            r_rc1 = -tf.expm1(-period / tau_rc)  # 1 - exp(-period/tau_rc)
            r_s1 = -tf.expm1(-period / tau_s)  # 1 - exp(-period/tau_s)
            return (1./d) * (q_rc/r_rc1 - q_s/r_s1)


    class AlphaRCNoiseBuilder(RCNoiseBuilder):
        @staticmethod
        def tensorflow(period, tau_s, tau_rc):
            d = tau_rc - tau_s
            u01 = tf.random_uniform(tf.shape(period))
            t = u01 * period
            q_rc = tf.exp(-t / tau_rc)
            q_s = tf.exp(-t / tau_s)
            r_rc1 = -tf.expm1(-period / tau_rc)  # 1 - exp(-period/tau_rc)
            r_s1 = -tf.expm1(-period / tau_s)  # 1 - exp(-period/tau_s)

            pt = tf.where(period < 100*tau_s, (period - t)*(1 - r_s1),
                          tf.zeros_like(period))
            qt = tf.where(t < 100*tau_s, q_s*(t + pt), tf.zeros_like(t))
            rt = qt / (tau_s * d * r_s1**2)
            rn = tau_rc*(q_rc/(d*d*r_rc1) - q_s/(d*d*r_s1)) - rt
            return rn


    NoiseBuilder.builders[type(None)] = NoNoiseBuilder
    NoiseBuilder.builders[LowpassRCNoise] = LowpassRCNoiseBuilder
    NoiseBuilder.builders[AlphaRCNoise] = AlphaRCNoiseBuilder


    class LoihiLIFBuilder(nengo_dl.neuron_builders.LIFBuilder):
        def __init__(self, ops, signals, config):
            super(LoihiLIFBuilder, self).__init__(ops, signals, config)

            self.spike_noise = NoiseBuilder.build(ops, signals, config)

        def _rate_step(self, J, dt):
            tau_ref = dt * tf.round(self.tau_ref / dt)
            tau_ref1 = tau_ref + 0.5*dt
            # ^ Since LoihiLIF takes `ceil(period/dt)` the firing rate is
            # always below the LIF rate. Using `tau_ref1` in LIF curve makes
            # it the average of the LoihiLIF curve (rather than upper bound).

            J -= self.one

            # --- compute Loihi rates (for forward pass)
            period = tau_ref + self.tau_rc*tf.log1p(tf.reciprocal(
                tf.maximum(J, self.epsilon)))
            period = dt * tf.ceil(period / dt)
            loihi_rates = self.spike_noise.generate(period, tau_rc=self.tau_rc)
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

                rates = self.amplitude / (tau_ref1 + self.tau_rc * q)
            else:
                rates = self.amplitude / (
                    tau_ref1 + self.tau_rc*tf.log1p(tf.reciprocal(
                        tf.maximum(J, self.epsilon))))
                rates = tf.where(J > self.zero, rates, self.zeros)

            # rates + stop_gradient(loihi_rates - rates) =
            #     loihi_rates on forward pass, rates on backwards
            return rates + tf.stop_gradient(loihi_rates - rates)

        def _step(self, J, voltage, refractory, dt):
            tau_ref = dt * tf.round(self.tau_ref / dt)

            delta_t = tf.clip_by_value(dt - refractory, self.zero, dt)
            voltage -= (J - voltage) * tf.expm1(-delta_t / self.tau_rc)

            spiked = voltage > self.one
            spikes = tf.cast(spiked, J.dtype) * self.alpha

            # refractory = tf.where(spiked, tau_ref, refractory - dt)
            refractory = tf.where(spiked,
                                  tau_ref + tf.zeros_like(refractory),
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
            tau_ref1 = 0.5*dt
            # ^ Since LoihiLIF takes `ceil(period/dt)` the firing rate is
            # always below the LIF rate. Using `tau_ref1` in LIF curve makes
            # it the average of the LoihiLIF curve (rather than upper bound).

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
