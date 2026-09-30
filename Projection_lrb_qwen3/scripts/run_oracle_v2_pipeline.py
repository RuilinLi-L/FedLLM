#!/usr/bin/env python3
from pathlib import Path
import argparse
import fcntl
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Projection_lrb_qwen3')]
from src.oracle_v2.protocol import load_config
from src.oracle_v2.pipeline import Pipeline

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT / 'Projection_lrb_qwen3/configs/oracle_v2.json'))
    p.add_argument('--gpus', default='4,5,6')
    p.add_argument('--stop-after', choices=['smoke', 'calibration', 'final'], default='final')
    a = p.parse_args()
    cfg = load_config(a.config)
    lock_path = Path(cfg['output_root']) / 'pipeline.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline = Pipeline(cfg, a.config, ROOT, [int(g) for g in a.gpus.split(',')])
        try:
            pipeline.run(a.stop_after)
        except Exception as e:
            pipeline.status('failed', error=str(e))
            raise
        finally:
            from src.oracle_v2.summary import summarize
            summarize(cfg['output_root'])
