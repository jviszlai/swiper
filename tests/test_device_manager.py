"""TODO: Test that DeviceManager schedules things properly. Specific cases to
test: 
    1) scheduling of "hanging" T operations (which need to be started the
        correct number of cycles before the relevant MERGE).
    2) scheduling of S correction does not happen when merge decoding is
        blocked, and does happen once it is unblocked.
    ...
"""
from swiper.schedule_experiments import *
from swiper.device_manager import DeviceManager, DeviceData

# def test_device_manager_simple():
