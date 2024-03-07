from .basic_controller import BasicMAC
from .ccdg_comm_controller import CCDGCommMAC

REGISTRY = {"basic_mac": BasicMAC,
            "ccdg_comm_mac": CCDGCommMAC}
