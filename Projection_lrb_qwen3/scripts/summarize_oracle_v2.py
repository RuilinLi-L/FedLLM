#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Projection_lrb_qwen3')]
from src.oracle_v2.protocol import load_config
from src.oracle_v2.summary import summarize
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT / 'Projection_lrb_qwen3/configs/oracle_v2.json'))
    a = p.parse_args()
    print(summarize(load_config(a.config)['output_root']))
