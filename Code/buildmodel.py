import distutils.dir_util
import os

from NetworkBasis import config as cfg
from NetworkBasis.networks import *

logs_path = os.path.join(cfg.path_result, cfg.experiment_name)

distutils.dir_util.mkpath(logs_path)

def buildmodel(inshape, write_summary=True):
    model = Network(inshape)

    if write_summary:
        with open(logs_path+'modelsummary.txt', 'w') as f:
            model.summary(line_length=140,print_fn=lambda x: f.write(x + '\n'))

    return model