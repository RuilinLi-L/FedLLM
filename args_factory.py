import argparse
import time
import sys

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='DAGER attack')

    parser.add_argument('--neptune', type=str, help='neptune project name, leave empty to not use neptune', default=None)
    parser.add_argument('--neptune_offline', action='store_true', help='Run Neptune in offline mode')
    parser.add_argument('--label', type=str, default='name of the run', required=False)
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Append all stdout/stderr to this UTF-8 file (Python streams only; use shell tee for raw FD capture).',
    )
    parser.add_argument('--log_append', action='store_true', help='Append to log_file instead of truncating.')
    
    # Method and setting
    parser.add_argument('--rng_seed', type=int, default=101) 
    parser.add_argument('--dataset', choices=['cola', 'sst2', 'rte', 'rotten_tomatoes', 'stanfordnlp/imdb', 'glnmario/ECHR'], required=True)
    parser.add_argument('--task', choices=['seq_class', 'next_token_pred'], required=True)
    parser.add_argument('--pad', choices=['right', 'left'], default='right')
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('-b','--batch_size', type=int, default=1)
    parser.add_argument('--n_inputs', type=int, required=True) # val:10/20, test:100
    parser.add_argument('--start_input', type=int, default=0)
    parser.add_argument('--end_input', type=int, default=100000)

    # Model path (defaults to huggingface download, use local path if offline)
    parser.add_argument('--model_path', type=str, default='bert-base-uncased')
    parser.add_argument('--finetuned_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--device_grad', type=str, default='cpu')
    parser.add_argument('--attn_implementation', type=str, default='sdpa', choices=['sdpa', 'eager'])

    parser.add_argument('--precision', type=str, default='full', choices=['8bit', 'half', 'full', 'double'])
    parser.add_argument('--parallel', type=int, default=1000)
    parser.add_argument('--grad_b', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--rank_tol', type=float, default=None) 
    parser.add_argument('--rank_cutoff', type=int, default=20)
    parser.add_argument('--l1_span_thresh', type=float, default=1e-5) 
    parser.add_argument('--l2_span_thresh', type=float, default=1e-3) 
    parser.add_argument('--l1_filter', choices=['maxB', 'all'], required=True)
    parser.add_argument('--l2_filter', choices=['overlap', 'non-overlap'], required=True)
    parser.add_argument('--distinct_thresh', type=float, default=0.7)
    parser.add_argument('--max_ids', type=int, default=-1)
    parser.add_argument('--maxC', type=int, default=10000000) 
    parser.add_argument('--reduce_incorrect', type=int, default=0)
    parser.add_argument('--n_incorrect', type=int, default=None)
    
    # FedAVG
    parser.add_argument('--algo', type=str, default='sgd', choices=['sgd', 'fedavg'])
    parser.add_argument('--avg_epochs', type=int, default=None)
    parser.add_argument('--b_mini', type=int, default=None)
    parser.add_argument('--avg_lr', type=float, default=None)
    parser.add_argument('--dist_norm', type=str, default='l2', choices=['l1', 'l2'])
    
    #DP
    parser.add_argument('--defense_noise', type=float, default=None) # add noise to true grads
    parser.add_argument('--max_len', type=int, default=1e10) 
    parser.add_argument('--p1_std_thrs', type=float, default=5)
    parser.add_argument('--l2_std_thrs', type=float, default=5)
    parser.add_argument('--dp_l2_filter', type=str, default='maxB', choices=['maxB', 'outliers'])
    parser.add_argument('--defense_pct_mask', type=float, default=None) # mask some percentage of gradients

    # Unified defense baselines (FL-LLM.md)
    parser.add_argument(
        '--defense',
        type=str,
        default='none',
        choices=['none', 'noise', 'dpsgd', 'topk', 'compression', 'soteria', 'mixup', 'dager', 'lrb'],
        help='Defense applied to client gradients before attack reconstruction; some defenses generate gradients directly.',
    )
    parser.add_argument(
        '--defense_clip_norm',
        type=float,
        default=1.0,
        help='Per-example L2 clip norm C for dpsgd.',
    )
    parser.add_argument(
        '--defense_topk_ratio',
        type=float,
        default=0.1,
        help='Fraction of |gradient| elements to keep per tensor (top-k sparsification).',
    )
    parser.add_argument(
        '--defense_n_bits',
        type=int,
        default=8,
        help='Bits per tensor for uniform gradient quantization (compression).',
    )
    parser.add_argument(
        '--defense_soteria_pruning_rate',
        type=float,
        default=60.0,
        help='Percent of classifier-input representation dimensions pruned by Soteria.',
    )
    parser.add_argument(
        '--defense_soteria_sample_dims',
        type=int,
        default=None,
        help='If set, score only this many random hidden dims (faster for large LLMs).',
    )
    parser.add_argument(
        '--defense_mixup_alpha',
        type=float,
        default=1.0,
        help='Beta(alpha, alpha) for MixUp mixing coefficient.',
    )
    parser.add_argument(
        '--defense_lrb_sensitive_n_layers',
        type=int,
        default=2,
        help='How many earliest transformer layers receive the strongest LRB protection.',
    )
    parser.add_argument(
        '--defense_lrb_keep_ratio_sensitive',
        type=float,
        default=0.2,
        help='Target low-resolution keep ratio for sensitive layers under LRB.',
    )
    parser.add_argument(
        '--defense_lrb_keep_ratio_other',
        type=float,
        default=0.75,
        help='Target low-resolution keep ratio for less-sensitive layers under LRB.',
    )
    parser.add_argument(
        '--defense_lrb_clip_scale_sensitive',
        type=float,
        default=0.5,
        help='Layer clip threshold multiplier (relative to median grad norm) for sensitive layers.',
    )
    parser.add_argument(
        '--defense_lrb_clip_scale_other',
        type=float,
        default=1.0,
        help='Layer clip threshold multiplier (relative to median grad norm) for less-sensitive layers.',
    )
    parser.add_argument(
        '--defense_lrb_noise_sensitive',
        type=float,
        default=0.03,
        help='Orthogonal noise multiplier for sensitive layers under LRB.',
    )
    parser.add_argument(
        '--defense_lrb_noise_other',
        type=float,
        default=0.005,
        help='Orthogonal noise multiplier for less-sensitive layers under LRB.',
    )
    parser.add_argument(
        '--defense_lrb_empirical_weight',
        type=float,
        default=0.6,
        help='Blend weight for on-the-fly gradient calibration in LRB (0=rule-only, 1=empirical-only).',
    )
    parser.add_argument(
        '--defense_lrb_calibration_samples',
        type=int,
        default=4096,
        help='Max elements per tensor used by LRB calibration sketches.',
    )
    parser.add_argument(
        '--defense_lrb_projection',
        type=str,
        default='signed_pool',
        choices=['signed_pool', 'pool'],
        help='Public subspace projection used by LRB; signed_pool is the randomized-basis default.',
    )
    
    # DAGER-specific defense parameters
    parser.add_argument(
        '--defense_dager_basis_perturb',
        action='store_true',
        default=True,
        help='Enable dynamic basis perturbation for DAGER defense.',
    )
    parser.add_argument(
        '--no_defense_dager_basis_perturb',
        action='store_false',
        dest='defense_dager_basis_perturb',
        help='Disable dynamic basis perturbation for DAGER defense.',
    )
    parser.add_argument(
        '--defense_dager_basis_noise_scale',
        type=float,
        default=0.01,
        help='Noise scale for dynamic basis perturbation.',
    )
    parser.add_argument(
        '--defense_dager_offset_embedding',
        action='store_true',
        default=False,
        help='Enable stochastic offset embedding for DAGER defense.',
    )
    parser.add_argument(
        '--defense_dager_offset_scale',
        type=float,
        default=0.01,
        help='Offset scale for stochastic offset embedding.',
    )
    parser.add_argument(
        '--defense_dager_gradient_slicing',
        action='store_true',
        default=False,
        help='Enable gradient slicing for DAGER defense.',
    )
    parser.add_argument(
        '--defense_dager_slice_first_n',
        type=int,
        default=None,
        help='Send only first n layers (if specified).',
    )
    parser.add_argument(
        '--defense_dager_slice_last_n',
        type=int,
        default=None,
        help='Send only last n layers (if specified).',
    )
    parser.add_argument(
        '--defense_dager_random_slice',
        action='store_true',
        default=False,
        help='Randomly select layers to send.',
    )
    parser.add_argument(
        '--defense_dager_slice_prob',
        type=float,
        default=0.5,
        help='Probability of sending each layer (if random_slice=True).',
    )
    parser.add_argument(
        '--defense_dager_rank_limit',
        action='store_true',
        default=False,
        help='Enable rank-limiting defense for DAGER.',
    )

    #Dropout
    parser.add_argument('--grad_mode', type=str, default='eval', choices=['eval', 'train'])
    
    #Rebuttal experiments
    parser.add_argument('--hidden_act', type=str, default=None)
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'mse'])
    
    #LoRA
    parser.add_argument('--train_method', type=str, default='full', choices=['full', 'lora'])
    parser.add_argument('--lora_r', type=int, default=None)

    if argv is None:
       argv = sys.argv[1:]
    args=parser.parse_args(argv)

    if args.log_file:
        from utils.terminal_log import install_terminal_log
        banner_argv = list(argv) if argv is not None else sys.argv[1:]
        install_terminal_log(
            args.log_file, append=args.log_append, argv_for_banner=banner_argv
        )

    if args.n_incorrect is None:
        args.n_incorrect = args.batch_size

    if args.neptune is not None:
        import neptune.new as neptune
        assert('label' in args)
        nep_par = { 'project':f"{args.neptune}", 'source_files':["*.py"] } 
        if args.neptune_offline:
            nep_par['mode'] = 'offline'
            args.neptune_id = 'DAG-0'

        run = neptune.init( **nep_par )
        args_dict = vars(args)
        run[f"parameters"] = args_dict
        args.neptune = run
        if not args.neptune_offline:
            print('waiting...')
            start_wait=time.time()
            args.neptune.wait()
            print('waited: ',time.time()-start_wait)
            args.neptune_id = args.neptune['sys/id'].fetch()
        print( '\n\n\nArgs:', *argv, '\n\n\n' ) 
    return args
