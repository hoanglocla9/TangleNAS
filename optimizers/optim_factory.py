from optimizers.mixop.gdas import GDASMixOp, GDASMixOpV2
from optimizers.mixop.darts import DARTSMixOp, DARTSMixOpV2
from optimizers.mixop.drnas import DRNASMixOp, DRNASMixOpV2
from optimizers.mixop.spos import SPOSMixOp
from optimizers.mixop.discrete import DiscretizeMixOp
from optimizers.sampler.darts import DARTSSampler
from optimizers.sampler.drnas import DRNASSampler
from optimizers.sampler.gdas import GDASSampler
from optimizers.sampler.spos import SPOSSampler


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
            return DiscretizeMixOp()
    else:
        if opt_name in ["darts_v1", "darts_v2"]:
            return DARTSMixOpV2()
        elif opt_name == "gdas":
            return GDASMixOpV2()
        elif opt_name == "drnas":
            return DRNASMixOpV2()
        else:
            raise NotImplementedError(f'WE v2 is not implemented for {opt_name}')

def get_sampler(opt_name):
    if opt_name in ["darts_v1", "darts_v2"]:
        return DARTSSampler()
    elif opt_name == "gdas":
        return GDASSampler()
    elif opt_name == "drnas":
        return DRNASSampler()
    elif opt_name == "spos":
        return SPOSSampler()
