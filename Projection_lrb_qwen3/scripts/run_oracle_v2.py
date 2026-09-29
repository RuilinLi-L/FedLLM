#!/usr/bin/env python3
"""Execute an explicit, immutable v2 job or prepare the versioned manifests."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Projection_lrb_qwen3')]
from src.oracle_v2.protocol import load_config, preregister, read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT / 'Projection_lrb_qwen3/configs/oracle_v2.json'))
    p.add_argument('--preregister', action='store_true')
    p.add_argument('--job')
    args = p.parse_args()
    config = load_config(args.config)
    if args.preregister:
        print(preregister(config, ROOT)['identity_sha256'], flush=True)
    elif args.job:
        from src.oracle_v2.worker import run_job
        run_job(config, read_json(args.job))
    else:
        p.error('Specify --preregister or --job')
if __name__ == '__main__':
    main()
