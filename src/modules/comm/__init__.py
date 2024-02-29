from .information_bottleneck_comm import IBComm
from .information_bottleneck_comm_full import IBFComm
from .information_bottleneck_comm_not_IB import IBNIBComm
from .ccdg_comm import CCDGComm
from .information_bottleneck_pruned_comm import IBPComm

REGISTRY = {"information_bottleneck": IBComm,
            "information_bottleneck_full": IBFComm,
            "information_bottleneck_not_IB": IBNIBComm,
            "ccdg": CCDGComm,
            "information_bottleneck_pruned": IBPComm}
