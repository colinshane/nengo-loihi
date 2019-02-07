from nengo_loihi.hardware.nxsdk_objects import LoihiSpikeInput


def test_strings():
    axon = LoihiSpikeInput.LoihiAxon(3, 5, 6, atom=8)
    assert str(axon) == "LoihiAxon(chip_id=3, core_id=5, axon_id=6, atom=8)"

    spike = LoihiSpikeInput.LoihiSpike(4, axon)
    assert str(spike) == (
        "LoihiSpike(time=4, chip_id=3, core_id=5, axon_id=6, atom=8)")
