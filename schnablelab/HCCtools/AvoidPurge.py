# -*- coding: UTF-8 -*-

"""
create, submit, canceal jobs. 
Find more details at HCC document: 
<https://hcc-docs.unl.edu>
"""

import numpy as np
import random
import os
import os.path as op
import sys
from schnablelab.apps.base import ActionDispatcher, OptionParser, glob,iglob
from schnablelab.apps.natsort import natsorted
from schnablelab.apps.header import Slurm_header
from subprocess import call
from subprocess import Popen
import subprocess

def main():
    actions = (
        ('action1', 'list, open, read, close files in dirs under purge policy'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def action1(args):
    """
    %prog dir

    do some tricky actions...
    """
    p = OptionParser(action1.__doc__)
    p.add_option("--num", default='10',
                 help="one num-th files will be read.")
    opts, args = p.parse_args(args)
    if len(args) != 1:
        sys.exit(not p.print_help())

    folder, = args
    all_fns = []
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            fn = os.path.join(dirpath, filename)
            all_fns.append(fn)
    part_fns = random.sample(all_fns, int(np.ceil(len(all_fns)/float(opts.num))))
    for i in part_fns:
        print(i)
        f = open(fn)
        f.readline()
        f.close()
    print('run away from crim scene !!!')

if __name__ == "__main__":
    main()
