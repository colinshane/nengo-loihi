import logging

from nengo.exceptions import ValidationError
import numpy as np

from nengo_loihi.discretize import (
    tracing_mag_int_frac,
    vth_to_manexp,
)
from nengo_loihi.hardware.nxsdk_objects import (
    Board,
    CxProfile,
    TraceCfg,
    VthProfile,
)

logger = logging.getLogger(__name__)


def compute_profiles(core, list_profiles):
    profile_lists = []
    for block in core.blocks:
        profile_lists.append(list_profiles(block))

    profiles = list(set(p for plist in profile_lists for p in plist))
    profile_idxs = {}
    for block, plist in zip(core.blocks, profile_lists):
        profile_idxs[block] = np.zeros(len(plist), dtype=int)
        for k, profile in enumerate(profiles):
            profile_idxs[block][[p == profile for p in plist]] = k

    return profiles, profile_idxs


def core_cx_profiles(core):
    """Compute all cxProfiles needed for a core"""
    def list_cx_profiles(block):
        profiles = []
        for i in range(block.compartment.n_compartments):
            profiles.append(CxProfile(
                decayU=block.compartment.decayU[i],
                decayV=block.compartment.decayV[i],
                refractDelay=block.compartment.refractDelay[i],
                enableNoise=block.compartment.enableNoise[i],
            ))

        return profiles

    return compute_profiles(core, list_cx_profiles)


def core_vth_profiles(core):
    """Compute all vthProfiles needed for a core"""
    def list_vth_profiles(block):
        profiles = []
        vth, _ = vth_to_manexp(block.compartment.vth)
        for i in range(block.compartment.n_compartments):
            profiles.append(VthProfile(
                vth=vth[i],
            ))

        return profiles

    return compute_profiles(core, list_vth_profiles)


def core_stdp_pre_cfgs(core):
    profiles = []
    profile_idxs = {}
    for synapse in core.synapses:
        if synapse.learning:
            mag_int, mag_frac = tracing_mag_int_frac(synapse.tracing_mag)
            tracecfg = TraceCfg(
                tau=synapse.tracing_tau,
                spikeLevelInt=mag_int,
                spikeLevelFrac=mag_frac,
            )

            if tracecfg in profiles:
                profile_idxs[synapse] = profiles.index(tracecfg)
            else:
                profile_idxs[synapse] = len(profiles)
                profiles.append(tracecfg)
        else:
            profile_idxs[synapse] = None

    return profiles, profile_idxs


class Allocator:
    """Responsible for allocating the board's devices to models."""

    def __call__(self, model):
        """Returns a Board object corresponding to the given model."""
        raise NotImplementedError()


class OneToOne(Allocator):
    """Assigns each block and input to distinct cores on the same chip."""

    def block_to_chip(self, block, chip):
        if block.compartment.n_compartments > 1024:
            raise ValidationError("Segment does not fit on one chip",
                                  "n_neurons")

        core = chip.new_core()
        core.add_block(block)

        cx_profiles, cx_profile_idxs = core_cx_profiles(core)
        [core.add_cx_profile(cx_profile) for cx_profile in cx_profiles]
        core.cx_profile_idxs = cx_profile_idxs

        vth_profiles, vth_profile_idxs = core_vth_profiles(core)
        [core.add_vth_profile(vth_profile) for vth_profile in vth_profiles]
        core.vth_profile_idxs = vth_profile_idxs

        for synapse in block.synapses:
            core.add_synapse(synapse)

        stdp_pre_cfgs, stdp_pre_cfg_idxs = core_stdp_pre_cfgs(core)
        [core.add_stdp_pre_cfg(stdp_pre_cfg) for stdp_pre_cfg in stdp_pre_cfgs]
        core.stdp_pre_cfg_idxs = stdp_pre_cfg_idxs

        core.stdp_pre_profile_idx = None  # hardware.builder will set
        core.stdp_profile_idx = None  # hardware.builder will set

    def input_to_chip(self, input, chip):
        # TODO: how to allocate inputs?
        core = chip.new_core()
        core.add_input(input)

    def __call__(self, model):
        board = Board()
        chip = board.new_chip()

        for block in model.blocks:
            self.block_to_chip(block, chip)

        for input in model.inputs:
            self.input_to_chip(input, chip)

        return board


class RoundRobin(OneToOne):
    """Assigns each block and input to the next chip in round-robin order."""

    def __init__(self, n_chips):
        self.n_chips = n_chips

    def __call__(self, model):
        board = Board()

        # We must dynamically allocate the chips
        # as needed because nxsdk==0.8.0 hits
        # an assertion if any chips contain 0 cores
        def get_chip(i):
            if i < self.n_chips:
                board.new_chip()
            return board.chips[i % self.n_chips]

        i = 0
        for block in model.blocks:
            self.block_to_chip(block, get_chip(i))
            i += 1

        for input in model.inputs:
            self.input_to_chip(input, get_chip(i))
            i += 1

        logger.info("Round-robin allocation across %d chips", board.n_chips)

        return board
