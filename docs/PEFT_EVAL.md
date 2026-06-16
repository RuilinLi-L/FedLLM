# PEFT / LoRA Evaluation And Training Notes

## PEFT Leakage Paths

`attack.py` and `scripts/peft_eval.sh` remain the DAGER-based PEFT span evaluation route.

The reportable PEFTLeak reproduction is image-side:

```bash
python attack_peftleak_image.py --mode vit_adapter \
  --dataset cifar100 \
  --data_root ./models_cache \
  --model_path torchvision_vit_small
```

This entrypoint reports `attack=peftleak_image_repro`. `--mode synthetic_ratio` is kept only for fast semantic/debug checks of the adapter gradient-ratio kernel.

The non-DAGER FedLLM PEFT text adaptation is separate:

```bash
bash scripts/peftleak_eval.sh sst2 2 bert-base-uncased 1 \
  --peft_method lora \
  --finetuned_path ./models/bert_sst2_lora_r16/final_adapter \
  --defense none
```

This entrypoint reports `attack=fedllm_peft_text_opt` and optimizes dummy input embeddings to match shared LoRA/IA3 adapter gradients. It does not use DAGER span decomposition and should not be reported as the original PEFTLeak reproduction.

The supported FedLLM PEFT text defense matrix is:

```text
none, noise, dpsgd, topk, compression, soteria, mixup, lrb, lrbprojonly, signed_bottleneck
```

`dager` is intentionally excluded from the FedLLM PEFT text matrix because it is DAGER-specific. If `attack_peftleak.py` receives `--defense dager`, it emits an explicit `unsupported` summary instead of folding that run into the text matrix.

The original image-side PEFTLeak reproduction lives in `docs/PEFTLEAK_REPRO.md` and `attacks/peftleak_image/`.

> Current training-defense status: LoRA/IA3 PEFT training supports
> `none / noise / topk / compression / lrb / lrbprojonly` plus
> `dpsgd / soteria / mixup / dager`. `dpsgd` is DP-SGD-style clipping plus
> Gaussian noise without a privacy accountant; `soteria` is a
> representation-masking style baseline; `mixup` is a manifold MixUp-style
> baseline and falls back to ordinary gradients when `batch_size < 2`.
> Prefix PEFT training still supports only post-gradient defenses; prefix
> direct-generation and DAGER training defenses remain unsupported. PEFT
> adapter-only gradients usually do not include position-embedding tensors, so
> `defense_dager_offset_embedding` is typically a no-op for LoRA/IA3 training
> unless position embeddings are trainable/shared.
>
> V1 eval scope: only `lora` and `ia3` are included in DAGER/partial-gradient
> PEFT evaluation tables. `prefix` is training/smoke-only in v1 and must not be
> mixed into PEFT privacy tables; Houlsby-style `adapter` remains v2 planned.

鏈枃妗ｈ鏄庡綋鍓嶄粨搴撲腑 PEFT 璺嚎鐨勫畾浣嶃€佹敮鎸佽寖鍥淬€佽缁?璇勬祴鍏ュ彛鍜屽疄楠岀煩闃点€傚綋鍓嶇増鏈槸 v1锛氫富绾夸粛鏄?Projection-LRB锛屼絾宸茬粡鎶?LoRA-only 鍏ュ彛鎵╁睍涓洪€氱敤 PEFT 鍏ュ彛銆?

## 1. 褰撳墠瀹氫綅

褰撳墠妗嗘灦鏀寔锛?

- PEFT DAGER eval锛歚lora / ia3`
- PEFT training锛歚lora / ia3 / prefix`锛屾敮鎸佽缁冩湡 post-gradient `none / noise / topk / compression / lrb / lrbprojonly`锛汱oRA/IA3 棰濆鏀寔璁粌鏈?`dpsgd / soteria / mixup / dager`
- BERT PEFT锛歚bert-base-uncased` 鍙敤浜?LoRA/IA3/Prefix 鐨?seq_class 璺嚎
- GPT-2 LoRA 涓荤嚎淇濇寔鍏煎锛孡lama LoRA 涓荤嚎淇濇寔鍏煎
- representation-side bottleneck v1锛氬湪 seq_class 鐨?classifier-input representation 涓婃墽琛?`mask / dropout / projection`
- legacy LoRA锛氭棫鐨?`--train_method lora` 浼氳嚜鍔ㄦ槧灏勫埌 `--train_method peft --peft_method lora`

褰撳墠鏆備笉鏀寔锛?

- 瀹屾暣 Houlsby-style Adapter銆俙--peft_method adapter` 浼氭槑纭姤閿欙細`adapter is planned but not enabled in v1`
- Prefix 璁粌鏈?DP-SGD-style / Soteria-style / MixUp-style / DAGER defense銆侺oRA/IA3 宸叉帴閫氳缁冩湡 baseline锛屼絾 `dpsgd` 娌℃湁 privacy accountant锛屼笉澹版槑 formal DP guarantee
- Prefix DAGER span eval銆侾refix 鍙互璁粌/smoke锛屼絾褰撳墠 DAGER eval 鍏ュ彛浼氭嫆缁?`--peft_method prefix`
- Llama Prefix銆倂1 鍙妸 Prefix smoke 璺嚎鏀惧湪 BERT/GPT-2

## 2. CLI 璇箟

鏂板叆鍙ｏ細

```bash
--train_method peft
--peft_method lora|ia3|prefix
```

鍏煎鍏ュ彛锛?

```bash
--train_method lora
```

浼氳嚜鍔ㄧ瓑浠蜂负锛?

```bash
--train_method peft --peft_method lora
```

DAGER PEFT eval 鍏ュ彛褰撳墠鍙敮鎸侊細

```bash
--train_method peft
--peft_method lora|ia3
```

LoRA 浠嶉渶瑕侊細

```bash
--lora_r 16
```

Prefix 鍙€夛細

```bash
--peft_num_virtual_tokens 20
```

representation bottleneck锛?

```bash
--defense_rep_bottleneck none|mask|dropout|projection
--defense_rep_keep_ratio 0.5
--defense_rep_dropout_p 0.1
```

## 3. PEFT 榛樿妯″潡

LoRA 榛樿 target modules锛?

| Model | Default |
|---|---|
| `gpt2`, `openai-community/gpt2-large` | `c_attn` |
| `bert-base-uncased` | `query,value` |
| Llama family | `q_proj` |

IA3 榛樿 target/feedforward modules锛?

| Model | Target modules | Feedforward modules |
|---|---|---|
| GPT-2 | `c_attn,c_fc` | `c_fc` |
| BERT | `query,value,intermediate.dense` | `intermediate.dense` |
| Llama | `q_proj,v_proj,down_proj` | `down_proj` |

Prefix 榛樿锛?

| Model | Default |
|---|---|
| GPT-2 | `num_virtual_tokens=20` |
| BERT | `num_virtual_tokens=20` |
| Llama | v2 planned |

## 4. Training

BERT LoRA smoke锛?

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path bert-base-uncased \
  --train_method peft \
  --peft_method lora \
  --lora_r 16 \
  --batch_size 2 \
  --num_epochs 1 \
  --models_cache ./models_cache \
  --output_dir ./models/bert_sst2_lora_r16
```

BERT IA3 smoke锛?

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path bert-base-uncased \
  --train_method peft \
  --peft_method ia3 \
  --batch_size 2 \
  --num_epochs 1 \
  --models_cache ./models_cache \
  --output_dir ./models/bert_sst2_ia3
```

BERT Prefix smoke锛?

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path bert-base-uncased \
  --train_method peft \
  --peft_method prefix \
  --peft_num_virtual_tokens 20 \
  --batch_size 2 \
  --num_epochs 1 \
  --models_cache ./models_cache \
  --output_dir ./models/bert_sst2_prefix
```

璁粌鏈?Projection-LRB锛?

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path bert-base-uncased \
  --train_method peft \
  --peft_method lora \
  --lora_r 16 \
  --defense lrbprojonly \
  --defense_lrb_keep_ratio_sensitive 0.5 \
  --batch_size 2 \
  --num_epochs 1
```

representation-side bottleneck锛?

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path bert-base-uncased \
  --train_method peft \
  --peft_method lora \
  --lora_r 16 \
  --defense none \
  --defense_rep_bottleneck projection \
  --defense_rep_keep_ratio 0.5 \
  --batch_size 2 \
  --num_epochs 1
```

鑴氭湰鍏ュ彛锛?

```bash
bash scripts/train_peft.sh sst2 2 bert-base-uncased lora --lora_r 16 --num_epochs 1
bash scripts/train_peft.sh sst2 2 bert-base-uncased ia3 --num_epochs 1
bash scripts/train_peft.sh sst2 2 bert-base-uncased prefix --peft_num_virtual_tokens 20 --num_epochs 1
```

## 5. Evaluation

鍗曟 PEFT DAGER eval锛?

```bash
bash scripts/peft_eval.sh sst2 2 bert-base-uncased 1 \
  --peft_method lora \
  --finetuned_path ./models/bert_sst2_lora_r16/final_adapter \
  --defense none
```

`--peft_method` 鐪佺暐鏃堕粯璁ゆ槸 `lora`銆侱AGER PEFT eval 褰撳墠鍙敮鎸?`lora / ia3`锛沗prefix` adapter 鍙互璁粌锛屼絾褰撳墠 span eval 浼氭槑纭嫆缁濄€侾EFT adapter 鐩綍浼氳鍙?`adapter_config.json`锛屽苟鏍￠獙锛?

- `peft_type`
- `target_modules`
- `feedforward_modules`
- `num_virtual_tokens`
- `task_type`
- `base_model_name_or_path`

LoRA legacy `.pt/.pth` checkpoint 浠嶆敮鎸侊紝浣嗗彧閫傜敤浜?LoRA锛屽苟涓斿繀椤绘樉寮忔彁渚?`--lora_r`銆?

## 6. 鎺ㄨ崘 v1 瀹為獙鐭╅樀

鏈€灏忕煩闃碉細

V1 PEFT eval matrix scope:

- Include in DAGER/partial-gradient eval tables: `peft_method=lora`, `peft_method=ia3`.
- Exclude from v1 eval tables: `peft_method=prefix`; keep it only as a training/smoke route.
- Exclude from current paper experiments: Houlsby-style `adapter`; it is v2 planned.

```text
dataset: sst2
model: bert-base-uncased
peft_method: lora / ia3
```

姣忕粍寤鸿璺戯細

```text
none
topk@0.1
compression@8
lrbprojonly@0.5
lrb full_lrb@0.5
rep_projection@0.5
rep_projection@0.5 + lrbprojonly@0.5
```

LoRA/GPT-2 涓荤嚎缁х画淇濈暀锛岀敤鏉ュ拰宸叉湁 DAGER/Projection-LRB 缁撴灉瀵归綈锛汢ERT 鐢ㄤ簬璇佹槑 encoder PEFT 娉涘寲銆?

## 7. 缁撴灉瀛楁

璁粌鍜?eval summary 浼氳褰曪細

- `train_method=peft`
- `peft_method`
- `peft_type`
- `peft_eval_scope` (`dager_eval`, `training_only`, `v2_planned`, or `n/a`)
- `peft_target_modules`
- `peft_feedforward_modules`
- `peft_num_virtual_tokens`
- `lora_r`
- `lora_target_modules`
- `rep_bottleneck_type`
- `rep_keep_ratio`
- `rep_dropout_p`
- `rep_bottleneck_with_lrb`

鏃у瓧娈典粛淇濈暀锛?

- `lora_checkpoint_type`
- `lora_adapter_r`
- `lora_adapter_target_modules`
- `lora_adapter_task_type`
- `lora_adapter_base_model`
- `lora_adapter_peft_type`

杩欐牱鏃ф棩蹇楄仛鍚堜笉浼氬潖锛屽悓鏃舵柊 PEFT 鏂规硶涓嶄細琚贩杩?LoRA-only 琛屻€?

## 8. 璁烘枃琛ㄨ堪杈圭晫

寤鸿璁烘枃涓繖鏍峰啓锛?

- Projection-LRB 鏄富鏂规硶
- IA3/BERT 鏄?DAGER PEFT eval 娉涘寲瀹為獙锛汸refix 淇濈暀涓鸿缁?smoke 璺嚎锛屽綋鍓嶄笉澹版槑 DAGER span eval 缁撴灉
- representation-side bottleneck 鏄?forward-side ablation锛屼笉澹版槑 formal DP
- LoRA/IA3 training-side `dpsgd / soteria / mixup` 鏄?baseline-style implementation锛屼笉澹版槑 formal DP 鎴栧師璁烘枃瀹屾暣澶嶇幇
- PEFT adapter-only gradients usually do not include position-embedding tensors, so `defense_dager_offset_embedding` is typically a no-op for LoRA/IA3 training unless position embeddings are trainable/shared.
- Adapter 鏄?v2 planned锛屼笉浣滀负褰撳墠瀹為獙缁撴灉澹扮О


