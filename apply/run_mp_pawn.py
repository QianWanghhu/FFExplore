"""Launches multiple PAWN experiments.

`mp_pawn` is launched naively by spinning up multiple experiements 
as separate processes via `subprocess`.

Quick and dirty approach to spreading the workload across available 
processors. 
"""

import argparse
import pathlib
from datetime import datetime
from glob import glob

import sys
import subprocess

from mp_pawn import mp_pawn


example_text = '''usage example:
python run_mp_pawn.py --sample_range 1600 2000 --step 100 --tuning 2 4 --ncores 4 --fdir [output directory]
'''

parser = argparse.ArgumentParser(description='pawn multiple PAWN processes',
                                 epilog=example_text,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--sample_range', type=int, nargs=2,
                    help='The range of samples to run over (start and end, inclusive)')
parser.add_argument('--step', type=int,
                    help='The incrementing step size')
parser.add_argument('--tuning', type=int, nargs='+',
                    default=[6, 8, 10, 12, 14],
                    help='The tuning values to use')
parser.add_argument('--ncores', type=int,
                    default=1,
                    help='Number of cores to use')
parser.add_argument('--fdir', type=str,
                    default=None,
                    help='Output directory')
parser.add_argument('--manager', type=bool, default=False, 
                    help='Is this the initial process or not? (default False)')


if __name__ == '__main__':
    args = parser.parse_args()

    s_start, s_end = args.sample_range
    step = args.step
    tuning = args.tuning
    ncores = args.ncores
    f_dir = args.fdir

    if not f_dir:
        from settings import PAWN_DATA_DIR
        f_dir = PAWN_DATA_DIR
    if not args.manager:
        mp_pawn(s_start, s_end, step, tuning, f_dir)
        sys.exit()

    for n_start in range(s_start, s_end, step):
        if n_start == (s_end - step):
            n_end = n_start + step + 1
        else:
            n_end = n_start + step - 1

        args = ['python', 'run_mp_pawn.py', '--sample_range', f'{n_start}', f'{n_end}', 
                '--step', f'{step}', '--tuning'] 
        args += [str(i) for i in tuning] + ['--fdir', f'{f_dir}']

        subprocess.Popen(args , creationflags=subprocess.DETACHED_PROCESS)
    

