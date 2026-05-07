import os
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from utils.ext import update_causal_mask
from utils.partial_models import add_partial_forward_gpt2, add_partial_forward_bert, add_partial_forward_llama
from utils.peft_utils import apply_lora_adapter
from constants import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from utils.functional import get_layer_decomp

def select_lora_gradient_indices(parameter_names, target_modules=None, preferred_modules=None):
    target_parts = []
    if target_modules and target_modules != 'n/a':
        target_parts = [
            part.strip().lower()
            for part in str(target_modules).split(',')
            if part.strip() and part.strip() != 'all-linear'
        ]
    preferred_parts = [
        part.strip().lower()
        for part in (preferred_modules or [])
        if str(part).strip()
    ]

    def is_adapter_name(name, *, prefer_a=False, preferred_only=False):
        lower = name.lower()
        if 'modules_to_save' in lower:
            return False
        if 'lora_' not in lower:
            return False
        if prefer_a and 'lora_a' not in lower:
            return False
        if target_parts and not any(part in lower for part in target_parts):
            return False
        if preferred_only and preferred_parts and not any(part in lower for part in preferred_parts):
            return False
        return True

    for prefer_a, preferred_only in (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ):
        selected = [
            (idx, name)
            for idx, name in enumerate(parameter_names)
            if is_adapter_name(name, prefer_a=prefer_a, preferred_only=preferred_only)
        ]
        if selected:
            return selected
    return []


class ModelWrapper():
    def __init__(self, args):
        assert (args.model_path in ['bert-base-uncased', 'gpt2', 'openai-community/gpt2-large', 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']),\
            'Model is not yet supported - add it to assertion list and specify implementation details'
        access_token = os.environ.get('HF_TOKEN')
        self.args = args
        self.full_model = None
        self.lora_gradient_indices = []
        self.lora_gradient_names = []
        model_kwargs = {'cache_dir': args.cache_dir} if args.cache_dir is not None else {}

        model_kwargs['pretrained_model_name_or_path'] = args.model_path if args.finetuned_path is None or args.train_method == 'lora' else args.finetuned_path
        model_kwargs['attn_implementation'] = args.attn_implementation

        if args.hidden_act is not None and args.model_path in ['gpt2', 'openai-community/gpt2-large']:
            model_kwargs['activation_function'] = args.hidden_act
        elif args.hidden_act is not None and args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']:
            model_kwargs['hidden_act'] = args.hidden_act
            
        if args.precision == '8bit':
            model_kwargs['load_in_8bit'] = True
        if args.precision == 'half':
            model_kwargs['torch_dtype'] = torch.float16
        if args.precision == 'double':
            model_kwargs['torch_dtype'] = torch.float64
        if args.task == 'seq_class':
            self.model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)
        elif args.task == 'next_token_pred':
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        else:
            assert False
        if args.task == 'seq_class' and args.finetuned_path is None:
            warnings.warn(
                (
                    'seq_class run without --finetuned_path: supported backbone models load a '
                    'newly initialized classifier head, so attack/defense results are not '
                    'trustworthy for baseline comparison.'
                ),
                stacklevel=2,
            )
        g_cpu = torch.Generator(device=self.model.device)
        g_cpu.manual_seed(0)
        self.model.eval()
        tokenizer_kwargs = {"use_fast": True, "cache_dir": args.cache_dir}
        if access_token:
            tokenizer_kwargs["token"] = access_token
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, **tokenizer_kwargs)
        self.tokenizer.model_max_length = 512
        
        if args.pad == 'left':
            self.tokenizer.padding_side = "left"
            
        if args.model_path in ['gpt2', 'openai-community/gpt2-large']:        
            self.start_token = None
            self.eos_token = self.model.config.eos_token_id
            self.layer_ids = list(range(4, 137, 12))
            
            if args.task == 'seq_class' and args.finetuned_path is None:
                self.model.score.weight.data.normal_( mean=0.0, std=1e-3, generator=g_cpu )
            
            # Set padding token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.pad_token = self.model.config.eos_token_id
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
            if args.train_method == 'lora':
                self.model = apply_lora_adapter(
                    self.model,
                    model_path=args.model_path,
                    lora_r=args.lora_r,
                    checkpoint_path=args.finetuned_path,
                    unwrap_base_model=False,
                    task=args.task,
                    target_modules=getattr(args, 'lora_target_modules', None),
                )
                self.full_model = self.model
                self.base_model = self._unwrap_model_for_attack(self.model)
            else:
                self.base_model = self.model

            self.embeddings_weight_nopos = self.base_model.transformer.wte.weight.unsqueeze(0)
            self.emb_size = self.base_model.config.n_embd
            add_partial_forward_gpt2(self.base_model.transformer)

        elif args.model_path in ['bert-base-uncased']:
            self.base_model = self.model
            self.start_token = 101
            self.eos_token = 102
            self.pad_token = 0
            self.layer_ids = list(range(5,190,16))
            
            # Store embeddings
            bert_embeddings_weight = self.model.bert.embeddings.word_embeddings.weight.unsqueeze(0)
            bert_embeddings_weight_token = self.model.bert.embeddings.token_type_embeddings.weight.unsqueeze(0)
            
            self.embeddings_weight_nopos = (bert_embeddings_weight_token + bert_embeddings_weight[0][:,None,:])[None,:,:,:]
            self.emb_size = self.model.config.hidden_size
            add_partial_forward_bert(self.model.bert)
        elif args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']:
            
            self.start_token = self.tokenizer.bos_token_id
            self.eos_token = self.tokenizer.eos_token_id
            if args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf']:
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.unk_token})
                self.pad_token = self.tokenizer.unk_token_id
                self.model.config.pad_token_id = self.tokenizer.unk_token_id
            else:
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
                self.pad_token = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            if args.train_method == 'lora' and args.finetuned_path is not None:
                self.model = apply_lora_adapter(
                    self.model,
                    model_path=args.model_path,
                    lora_r=args.lora_r,
                    checkpoint_path=args.finetuned_path,
                    unwrap_base_model=False,
                    task=args.task,
                    target_modules=getattr(args, 'lora_target_modules', None),
                )
                self.full_model = self.model
                self.base_model = self._unwrap_model_for_attack(self.model)
                self.layer_ids = list(range(0,64,2))
            else:
                if args.task == 'seq_class' and args.finetuned_path is None:
                    self.model.score.weight.data.normal_(mean=0.0, std=1e-3)
                #else:
                    #self.model.lm_head.weight.data.normal_(mean=0.0, std=1e-6)
                    
                if args.train_method == 'lora':
                    self.full_model = apply_lora_adapter(
                        self.model,
                        model_path=args.model_path,
                        lora_r=args.lora_r,
                        checkpoint_path=None,
                        unwrap_base_model=False,
                        task=args.task,
                        target_modules=getattr(args, 'lora_target_modules', None),
                    )
                    self.model = self.full_model
                    self.base_model = self._unwrap_model_for_attack(self.model)
                    self.layer_ids = list(range(1,64,2))

                else:
                    self.base_model = self.model
                    self.layer_ids = list(range(1,281,9))

            self.emb_size = self.base_model.config.hidden_size
            self.embeddings_weight_nopos = self.base_model.model.embed_tokens.weight.unsqueeze(0)
            add_partial_forward_llama(self.base_model.model)

        self.trainable_parameters = lambda: (param for param in self.model.parameters() if param.requires_grad)
        self.trainable_parameter_names = self._trainable_parameter_names
        if args.train_method == 'lora':
            self._refresh_lora_gradient_inventory()
        config['START_TOKEN'] = self.start_token
        config['EOS_TOKEN'] = self.eos_token
        config['PAD_TOKEN'] = self.pad_token
        self.set_model_device(args.device)

    def _unwrap_model_for_attack(self, model):
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            return model.base_model.model
        if hasattr(model, 'model') and not hasattr(model, 'transformer'):
            return model.model
        return model

    def _trainable_parameter_names(self):
        return [name for name, param in self.model.named_parameters() if param.requires_grad]

    def _refresh_lora_gradient_inventory(self):
        names = self._trainable_parameter_names()
        selected = select_lora_gradient_indices(
            names,
            getattr(self.args, 'lora_target_modules', None),
            preferred_modules=self._lora_span_preferred_modules(),
        )
        adapter_indices = [idx for idx, _ in selected]
        adapter_names = [name for _, name in selected]
        if not adapter_indices:
            raise ValueError(
                'LoRA trainable parameter inventory is empty. Expected lora_A/lora_B adapter '
                'parameters, excluding modules_to_save classifier heads.'
            )
        self.lora_gradient_indices = adapter_indices
        self.lora_gradient_names = adapter_names

    def _lora_span_preferred_modules(self):
        if self.args.model_path in ['gpt2', 'openai-community/gpt2-large']:
            return ['c_attn']
        if self.args.model_path in [
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-70b-hf',
            'meta-llama/Meta-Llama-3-8B',
            'meta-llama/Meta-Llama-3.1-8B',
            'meta-llama/Meta-Llama-3-70B',
        ]:
            return ['q_proj']
        return []

    def _classifier_model(self):
        if self.args.train_method == 'lora' and hasattr(self.model, 'score'):
            return self.model
        return self.base_model
        
    def compute_grads_fed_avg(self, batch, labels, create_graph=False):
        trainable_params = list(self.trainable_parameters())
        og_weights = [param.data.clone() for param in trainable_params]

        self.model.eval()
        optimizer = torch.optim.SGD(trainable_params, lr=self.args.avg_lr)

        n_minib = batch['input_ids'].shape[0] // self.args.b_mini
        print(n_minib)
        for _ in range(self.args.avg_epochs):
            for i in range(n_minib):
                print(batch['input_ids'].shape)
                b_mini = {k:batch[k][i*self.args.b_mini:(i+1)*self.args.b_mini] for k in batch.keys()}
                y_mini = labels[:, i*self.args.b_mini:(i+1)*self.args.b_mini]
                print(b_mini['input_ids'].shape, y_mini)
                optimizer.zero_grad()
                outs = self.model(**b_mini, labels=y_mini)
                outs.loss.backward()
                optimizer.step()
           
        grad = [-(param.data.detach() - og_weights[i])/n_minib/self.args.avg_lr/self.args.avg_epochs for i, param in enumerate(trainable_params)]
        for i, param in enumerate(trainable_params):
            param.data = og_weights[i]
        self.model.eval()
        return grad

    def _set_grad_mode(self):
        if self.args.grad_mode == 'eval':
            self.model.eval()
        else:
            self.model.train()

    def _prepare_grad_context(self, batch, y_labels):
        self._set_grad_mode()
        dev = y_labels.device
        if self.args.precision != '8bit':
            batch = batch.to(self.args.device_grad)
            y_labels = y_labels.to(self.args.device_grad)
            self.model.to(self.args.device_grad)
        return batch, y_labels, dev

    def _restore_grad_context(self, batch, y_labels, dev):
        self.set_model_device(dev)
        if self.args.precision != '8bit':
            batch = batch.to(dev)
            y_labels = y_labels.to(dev)
        self.model.eval()

    def _prepare_task_labels(self, batch, y_labels):
        if self.args.task == 'next_token_pred':
            return torch.where(batch['attention_mask'].bool(), batch['input_ids'], -100)
        if self.args.task == 'seq_class':
            return y_labels.view(-1).long()
        raise ValueError(f'Unsupported task: {self.args.task}')

    def _slice_batch(self, batch, idx):
        return {k: v[idx:idx+1] for k, v in batch.items()}

    def _compute_standard_grads_prepared(self, batch, labels, create_graph=False):
        self.model.zero_grad(set_to_none=True)
        if self.is_lower() and self.args.task == 'seq_class':
            outputs = self.model(**batch)
            logits = outputs.logits.float()
            loss = torch.nn.functional.nll_loss(
                torch.nn.functional.log_softmax(logits, dim=-1),
                labels.view(-1),
            )
            return torch.autograd.grad(
                loss,
                self.trainable_parameters(),
                create_graph=create_graph,
                allow_unused=True,
            )

        outs = self.model(**batch, labels=labels, output_hidden_states=True)
        if self.args.loss == 'mse':
            loss = outs.hidden_states[-1].pow(2).mean()
        elif self.args.loss == 'ce':
            loss = outs.loss
        else:
            raise ValueError(f'Unsupported loss: {self.args.loss}')
        return torch.autograd.grad(
            loss,
            self.trainable_parameters(),
            create_graph=create_graph,
            allow_unused=True,
        )

    def _seq_class_input_embeds(self, batch):
        if self.args.model_path in ['gpt2', 'openai-community/gpt2-large']:
            return self.base_model.transformer.wte(batch['input_ids'])

        if self.args.model_path in ['bert-base-uncased']:
            bert = self.model.bert
            el = bert.embeddings
            input_ids = batch['input_ids']
            token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids))
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            word = el.word_embeddings(input_ids)
            pos_e = el.position_embeddings(position_ids)
            tok_e = el.token_type_embeddings(token_type_ids)
            emb = el.LayerNorm(word + pos_e + tok_e)
            emb = el.dropout(emb)
            return emb

        if self.args.model_path in [
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-70b-hf',
            'meta-llama/Meta-Llama-3-8B',
            'meta-llama/Meta-Llama-3.1-8B',
            'meta-llama/Meta-Llama-3-70B',
        ]:
            return self.base_model.model.embed_tokens(batch['input_ids'])

        raise NotImplementedError(f'Seq-class embeddings not implemented for {self.args.model_path}')

    def _seq_class_representation_from_embeds(self, batch, inputs_embeds, representation_mask=None):
        model = self.base_model
        attn = batch.get('attention_mask')

        if self.args.model_path in ['gpt2', 'openai-community/gpt2-large']:
            position_ids = torch.arange(
                batch['input_ids'].size(1),
                device=batch['input_ids'].device,
            ).unsqueeze(0).expand_as(batch['input_ids'])
            tr_out = model.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                position_ids=position_ids,
            )
            hidden = tr_out.last_hidden_state
            if attn is not None:
                idx = attn.long().sum(dim=1) - 1
                idx = idx.clamp(min=0)
            else:
                idx = torch.full(
                    (hidden.size(0),),
                    hidden.size(1) - 1,
                    dtype=torch.long,
                    device=hidden.device,
                )
            representation = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
            if representation_mask is not None:
                representation = representation * representation_mask
            return representation

        if self.args.model_path in ['bert-base-uncased']:
            bert = model.bert
            input_ids = batch['input_ids']
            raw_mask = batch.get('attention_mask')
            ext_mask = bert.get_extended_attention_mask(raw_mask, input_ids.shape) if raw_mask is not None else None
            enc_out = bert.encoder(inputs_embeds, attention_mask=ext_mask)
            sequence_output = enc_out.last_hidden_state
            if bert.pooler is not None:
                representation = bert.pooler(sequence_output)
            else:
                representation = sequence_output[:, 0]
            if representation_mask is not None:
                representation = representation * representation_mask
            return representation

        if self.args.model_path in [
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-70b-hf',
            'meta-llama/Meta-Llama-3-8B',
            'meta-llama/Meta-Llama-3.1-8B',
            'meta-llama/Meta-Llama-3-70B',
        ]:
            llama = model.model
            input_ids = batch['input_ids']
            position_ids = torch.arange(
                input_ids.size(1),
                device=input_ids.device,
            ).unsqueeze(0).expand_as(input_ids)
            out = llama(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                position_ids=position_ids,
            )
            hidden = out.last_hidden_state
            if attn is not None:
                idx = attn.long().sum(dim=1) - 1
                idx = idx.clamp(min=0)
            else:
                idx = torch.full(
                    (hidden.size(0),),
                    hidden.size(1) - 1,
                    dtype=torch.long,
                    device=hidden.device,
                )
            representation = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
            if representation_mask is not None:
                representation = representation * representation_mask
            return representation

        raise NotImplementedError(f'Seq-class forward not implemented for {self.args.model_path}')

    def _seq_class_logits_from_representation(self, representation):
        if self.args.model_path in ['gpt2', 'openai-community/gpt2-large']:
            return self._classifier_model().score(representation)

        if self.args.model_path in ['bert-base-uncased']:
            return self.model.classifier(self.model.dropout(representation))

        if self.args.model_path in [
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-70b-hf',
            'meta-llama/Meta-Llama-3-8B',
            'meta-llama/Meta-Llama-3.1-8B',
            'meta-llama/Meta-Llama-3-70B',
        ]:
            return self._classifier_model().score(representation)

        raise NotImplementedError(f'Seq-class classifier head not implemented for {self.args.model_path}')

    def _seq_class_logits_from_embeds(self, batch, inputs_embeds, representation_mask=None):
        representation = self._seq_class_representation_from_embeds(
            batch,
            inputs_embeds,
            representation_mask=representation_mask,
        )
        logits = self._seq_class_logits_from_representation(representation)
        return logits, representation

    def compute_per_example_grads(self, batch, y_labels, create_graph=False, sample_grad_fn=None):
        if self.args.algo == 'fedavg':
            raise NotImplementedError('Per-example gradients are not implemented for algo=fedavg.')
        if self.args.grad_b is not None:
            raise NotImplementedError('Per-example gradients are not implemented with grad_b mini-batching.')

        batch, y_labels, dev = self._prepare_grad_context(batch, y_labels)
        try:
            labels = self._prepare_task_labels(batch, y_labels)
            grad_list = []
            for idx in range(batch['input_ids'].shape[0]):
                sample_batch = self._slice_batch(batch, idx)
                sample_labels = labels[idx:idx+1]
                if sample_grad_fn is None:
                    grad = self._compute_standard_grads_prepared(
                        sample_batch,
                        sample_labels,
                        create_graph=create_graph,
                    )
                else:
                    grad = sample_grad_fn(
                        sample_batch,
                        sample_labels,
                        sample_idx=idx,
                        create_graph=create_graph,
                    )
                grad_list.append(
                    tuple(
                        g if (g is None or create_graph) else g.detach().clone()
                        for g in grad
                    )
                )
            return grad_list
        finally:
            self._restore_grad_context(batch, y_labels, dev)

    def compute_grads(self, batch, y_labels, create_graph=False):
        batch, y_labels, dev = self._prepare_grad_context(batch, y_labels)
        try:
            labels = self._prepare_task_labels(batch, y_labels)
            if self.args.grad_b is None:
                if self.args.algo == 'fedavg':
                    if self.args.task != 'seq_class':
                        raise NotImplementedError('FedAvg gradients are only supported for seq_class.')
                    grad = self.compute_grads_fed_avg(batch, y_labels, create_graph)
                else:
                    grad = self._compute_standard_grads_prepared(batch, labels, create_graph=create_graph)
            else:
                if self.args.algo == 'fedavg':
                    raise NotImplementedError('grad_b mini-batching is not implemented for algo=fedavg.')
                self.model.zero_grad(set_to_none=True)
                minib_size = self.args.batch_size // self.args.grad_b
                for i in range(self.args.grad_b):
                    mini_batch = {
                        k: batch[k][i*minib_size:(i+1)*minib_size]
                        for k in batch.keys()
                    }
                    mini_labels = labels[i*minib_size:(i+1)*minib_size]
                    if self.is_lower() and self.args.task == 'seq_class':
                        outputs = self.model(**mini_batch)
                        logits = outputs.logits.float()
                        loss = torch.nn.functional.nll_loss(
                            torch.nn.functional.log_softmax(logits, dim=-1),
                            mini_labels.view(-1),
                        )
                        loss.backward()
                    else:
                        outs = self.model(**mini_batch, labels=mini_labels)
                        outs.loss.backward()
                grad = tuple([
                    param.grad.detach().cpu() / self.args.grad_b
                    for param in self.trainable_parameters()
                ])
            return grad
        finally:
            self._restore_grad_context(batch, y_labels, dev)

    def compute_grads_mixup(self, batch, y_labels, create_graph=False):
        """
        Representation-level manifold MixUp for seq_class.

        Falls back to standard gradients for non-seq_class tasks or batch_size < 2.
        """
        batch, y_labels, dev = self._prepare_grad_context(batch, y_labels)
        try:
            labels = self._prepare_task_labels(batch, y_labels)
            if self.args.task != 'seq_class':
                return self._compute_standard_grads_prepared(batch, labels, create_graph=create_graph)

            batch_size = batch['input_ids'].shape[0]
            if batch_size < 2:
                return self._compute_standard_grads_prepared(batch, labels, create_graph=create_graph)

            alpha = float(getattr(self.args, 'defense_mixup_alpha', 1.0))
            perm = torch.randperm(batch_size, device=batch['input_ids'].device)
            lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
            emb = self._seq_class_input_embeds(batch)
            representation = self._seq_class_representation_from_embeds(batch, emb)
            representation_mixed = lam * representation + (1.0 - lam) * representation[perm]
            logits = self._seq_class_logits_from_representation(representation_mixed)
            loss = lam * F.cross_entropy(logits, labels) + (1.0 - lam) * F.cross_entropy(logits, labels[perm])
            self.model.zero_grad(set_to_none=True)
            return torch.autograd.grad(
                loss,
                self.trainable_parameters(),
                create_graph=create_graph,
                allow_unused=True,
            )
        finally:
            self._restore_grad_context(batch, y_labels, dev)

    def set_model_device(self, device):
        if self.args.precision == '8bit':
            return
        if self.args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B'] and device!='cpu':
            self.base_model.model.embed_tokens.to(device)
            self.base_model.model.rotary_emb.to(device)
            for i in range(self.args.n_layers):
                self.base_model.model.layers[i].to(device)
        else:
            self.model.to(device)

    def get_matrices_expansions(self, true_grads, B=None, tol=None):
        if self.args.train_method == 'lora':
            grad_indices = self.lora_gradient_indices
            grad_names = self.lora_gradient_names
            if len(grad_indices) < self.args.n_layers:
                raise ValueError(
                    'LoRA DAGER span check needs at least args.n_layers adapter tensors; '
                    f'found {len(grad_indices)} tensor(s): {grad_names}. '
                    'Check --lora_target_modules and adapter_config.json.'
                )
        else:
            grad_indices = self.layer_ids
            grad_names = [str(idx) for idx in grad_indices]

        rank_cap = None
        for idx, name in zip(grad_indices[:self.args.n_layers], grad_names[:self.args.n_layers]):
            if idx >= len(true_grads):
                raise ValueError(
                    f'Gradient index {idx} for {name} is outside true_grads '
                    f'length {len(true_grads)}.'
                )
            grad = true_grads[idx]
            if grad is None:
                raise ValueError(f'Gradient tensor for {name} is None; cannot build DAGER span.')
            if grad.ndim < 2:
                raise ValueError(f'Gradient tensor for {name} must be matrix-like; got shape {tuple(grad.shape)}.')
            current_cap = min(int(grad.shape[-2]), int(grad.shape[-1]))
            rank_cap = current_cap if rank_cap is None else min(rank_cap, current_cap)
        if rank_cap is None or rank_cap <= 0:
            raise ValueError('No matrix-like gradients available for DAGER span decomposition.')

        if B is None:
            max_rank = 0
            for i in grad_indices[:10]:
                grad = true_grads[i]
                if self.args.precision == 'half':
                    B = np.linalg.matrix_rank( grad.float().cpu() , tol=tol)
                else:
                    B = np.linalg.matrix_rank( grad.cpu() , tol=tol)
                if max_rank < B:
                    max_rank = B
            B = max_rank
        if self.args.algo == 'fedavg':
            B += 60
        B = min(B, self.emb_size - self.args.rank_cutoff, rank_cap)
        if B <= 0:
            raise ValueError(
                f'DAGER span rank must be positive after clipping; got B={B}, '
                f'rank_cap={rank_cap}, emb_size={self.emb_size}, rank_cutoff={self.args.rank_cutoff}.'
            )
        
        R_Qs = []
        
        for i in range(self.args.n_layers):
            grad_Q = true_grads[grad_indices[i]]
            if self.args.train_method != 'lora' and self.args.model_path in ['gpt2', 'openai-community/gpt2-large']:
                grad_Q = grad_Q.T
            _, R_Q = get_layer_decomp(grad_Q, B=B, tol=tol, upcast=(self.args.precision=='half'))
            R_Q = R_Q.to(self.args.device)
            R_Qs.append(R_Q)
        return B, R_Qs
            
    def get_embeddings(self, pos = None):
        if self.args.model_path in ['bert-base-uncased']:
            bert_embeddings_weight_position = self.model.bert.embeddings.position_embeddings.weight.unsqueeze(0)
            emb = self.embeddings_weight_nopos.to(self.args.device) + bert_embeddings_weight_position[0][pos:pos+1,None,None,:]
            emb = self.model.bert.embeddings.LayerNorm(emb)
            return emb
        
        elif self.args.model_path in ['gpt2', 'openai-community/gpt2-large']:
            gpt_embeddings_weight_position = self.base_model.transformer.wpe.weight.unsqueeze(0)
            emb = self.embeddings_weight_nopos.to(self.args.device) + gpt_embeddings_weight_position[0][pos:pos+1,None,:]
            emb = self.base_model.transformer.h[0].ln_1(emb)
            return emb
        elif self.args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']:
            emb = self.embeddings_weight_nopos.to(self.args.device)
            return self.base_model.model.layers[0].input_layernorm(emb)
        
    def get_layer_inputs(self, sentences, token_type_ids=None, attention_mask=None, layers=1):
        if self.args.model_path in ['bert-base-uncased']:
            # if token_type_ids is None:
            #     raise ValueError('Token type must be defined when model is BERT')
            # emb = self.model.bert.embeddings( input_ids=sentences, token_type_ids=token_type_ids )
            # layer_inputs = []
            # for i in range(layers):
            #     emb = self.model.bert.encoder.layer[i](emb)[0]# As end of sentence tokens have little gradient they are unreliable measures for sentence inclusion
            #     layer_inputs.append(emb[ : , :-1, : ].clone())
            # return layer_inputs
            return self.model.bert.get_hidden_states(input_ids=sentences, token_type_ids=token_type_ids, n_layers=layers)
        
        elif self.args.model_path in ['gpt2', 'openai-community/gpt2-large']:
            return self.base_model.transformer.get_hidden_states(input_ids=sentences, attention_mask=attention_mask, n_layers=layers)
        
        elif self.args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']:
            position_ids = torch.arange(sentences.size(1)).unsqueeze(0).repeat(sentences.size(0), 1).to(self.args.device)
            # if attention_mask is not None:
            #     first_item_idx = torch.argmax(attention_mask, dim=1).unsqueeze(1)
            #     position_ids = torch.maximum(position_ids - first_item_idx, torch.tensor(0).to(self.args.device))
            #     attention_mask = update_causal_mask(self.model.model, attention_mask, emb).to(self.args.device)

            # layer_inputs = []
            # for i in range(layers):
            #     emb = self.model.model.layers[i](emb, attention_mask=attention_mask, position_ids=position_ids)[0]# As end of sentence tokens have little gradient they are unreliable measures for sentence inclusion
            #     layer_inputs.append(self.model.model.layers[i+1].input_layernorm(emb))
            # return layer_inputs
            return self.base_model.model.get_hidden_states(input_ids=sentences, position_ids=position_ids,attention_mask=attention_mask, n_layers=layers)
        
    def is_bert(self):
        return self.args.model_path in ['bert-base-uncased']
    
    def is_decoder(self):
        return self.args.model_path in ['gpt2', 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf','openai-community/gpt2-large', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']
        
    def has_rope(self):
        return self.args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']

    def has_bos(self):
        return self.start_token is not None
    def is_lower(self):
        return self.args.precision in ['8bit', 'half']
