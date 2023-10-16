from optimizers.mixop.gdas import GDASMixOp, GDASMixOpV2
from optimizers.mixop.darts import DARTSMixOp, DARTSMixOpV2
from optimizers.mixop.drnas import DRNASMixOp, DRNASMixOpV2
from optimizers.mixop.spos import SPOSMixOp, SPOSMixOpV2
from optimizers.mixop.discrete import DiscretizeMixOpV2
from optimizers.sampler.darts import DARTSSampler
from optimizers.sampler.drnas import DRNASSampler
from optimizers.sampler.gdas import GDASSampler
from optimizers.sampler.spos import SPOSSampler
from optimizers.sampler.discrete import DiscreteSampler


def get_mixop(opt_name, use_we_v2=False):
    if not use_we_v2:
        if opt_name in ["darts_v1", "darts_v2"]:
            return DARTSMixOp()
        elif opt_name == "gdas":
            return GDASMixOp()
        elif opt_name == "drnas":
            return DRNASMixOp()
        elif opt_name == "spos":
            return SPOSMixOp()
        elif opt_name == "discrete":
            return DiscretizeMixOpV2()
    else:
        if opt_name in ["darts_v1", "darts_v2"]:
            return DARTSMixOpV2()
        elif opt_name == "gdas":
            return GDASMixOpV2()
        elif opt_name == "drnas":
            return DRNASMixOpV2()
        elif opt_name == "spos":
            return SPOSMixOpV2()
        elif opt_name == "discrete":
            return DiscretizeMixOpV2()

def get_sampler(opt_name):
    if opt_name in ["darts_v1", "darts_v2"]:
        return DARTSSampler()
    elif opt_name == "gdas":
        return GDASSampler()
    elif opt_name == "drnas":
        return DRNASSampler()
    elif opt_name == "spos":
        return SPOSSampler()
    elif opt_name == "discrete":
        return DiscreteSampler()
    
