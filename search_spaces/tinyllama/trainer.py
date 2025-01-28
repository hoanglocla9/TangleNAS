# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib, functools
import warnings, time, os, shutil
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from dataclasses import dataclass, field


from architect import ArchitectV1

import torch, math
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils.data import Dataset, RandomSampler

from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, deepspeed_init, deepspeed_load_checkpoint
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.trainer import Trainer, _is_peft_model
from transformers.utils import is_datasets_available, logging, is_torch_xla_available, is_apex_available, is_accelerate_available, is_sagemaker_mp_enabled, is_safetensors_available, is_peft_available
from transformers.utils.deprecation import deprecate_kwarg
from transformers.trainer_utils import has_length, seed_worker
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.trainer_callback import (
    ExportableState,
    TrainerCallback,
    TrainerState,
)
import pickle
from transformers.integrations import (
    hp_params,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

from transformers.integrations.tpu import tpu_spmd_dataloader
from torch.utils.data import DataLoader

from packaging import version
import torch.distributed as dist
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_pt_utils import (
    get_model_param_count,
)
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel

if is_apex_available():
    from apex import amp
if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedType,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False

from torch.utils.data import IterableDataset
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, PredictionOutput, speed_metrics, TrainOutput, HPSearchBackend
from transformers.training_args import TrainingArguments, OptimizerNames, ParallelMode
from accelerate.utils import DistributedType
from transformers.optimization import get_scheduler

import numpy as np

logger = logging.get_logger(__name__)

class CosineSched:
    def __init__(self, start_step, max_step, eta_start, eta_end, **xargs):
        assert start_step < max_step

        self.start_step = start_step
        self.max_step = max_step
        self.eta_start = eta_start
        self.eta_end = eta_end
    
    def _step(self, cur_step):
        pass

    def __call__(self, cur_step):
        if cur_step < self.start_step:
            return self.eta_start
        
        eta_cur_step = self.eta_end + 1/2 * (self.eta_start - self.eta_end) * (np.cos(np.pi * \
                                                                                    (cur_step - self.start_step)/(self.max_step - self.start_step) \
                                                                                    ) + 1)
    
        return eta_cur_step
    
@dataclass
class SupernetTrainngArguments(TrainingArguments):
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `num_beams` value of the model configuration."
            )
        },
    )
    generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON
        serialization support). It obfuscates the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = super().to_dict()
        for k, v in d.items():
            if isinstance(v, GenerationConfig):
                d[k] = v.to_dict()
        return d


class SupernetTrainer(Trainer):
    @deprecate_kwarg("tokenizer", new_name="processing_class", version="5.0.0", raise_if_both_names=True)
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Union[Dataset, "IterableDataset", "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union["PreTrainedTokenizerBase", "BaseImageProcessor", "FeatureExtractionMixin", "ProcessorMixin"]
        ] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        self._created_lr_scheduler_q = False
        self.optimizer_q, self.lr_scheduler_q = None, None
        self.architect = ArchitectV1(model=self.model)

        # Override self.model.generation_config if a GenerationConfig is specified in args.
        # Priority: args.generation_config > model.generation_config > default GenerationConfig.
        if self.args.generation_config is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.model.generation_config = gen_config

    @staticmethod
    def load_generation_config(gen_config_arg: Union[str, GenerationConfig]) -> GenerationConfig:
        """
        Loads a `~generation.GenerationConfig` from the `Seq2SeqTrainingArguments.generation_config` arguments.

        Args:
            gen_config_arg (`str` or [`~generation.GenerationConfig]`):
                `Seq2SeqTrainingArguments.generation_config` argument.

        Returns:
            A `~generation.GenerationConfig`.
        """

        # GenerationConfig provided, nothing to do
        if isinstance(gen_config_arg, GenerationConfig):
            gen_config = deepcopy(gen_config_arg)
        else:
            # str or Path
            pretrained_model_name = Path(gen_config_arg) if isinstance(gen_config_arg, str) else gen_config_arg
            config_file_name = None

            # Figuring if it is path pointing to a file, pointing to a directory or else a model id or URL
            # This step is required in order to determine config_file_name
            if pretrained_model_name.is_file():
                config_file_name = pretrained_model_name.name
                pretrained_model_name = pretrained_model_name.parent
            # dir path
            elif pretrained_model_name.is_dir():
                pass
            # model id or URL
            else:
                pretrained_model_name = gen_config_arg

            gen_config = GenerationConfig.from_pretrained(pretrained_model_name, config_file_name)

        # Strict validation to fail early. `GenerationConfig.save_pretrained()`, run at the end of training, throws
        # an exception if there are warnings at validation time.
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                gen_config.validate()
            if len(caught_warnings) > 0:
                raise ValueError(str([w.message for w in caught_warnings]))
        except ValueError as exc:
            raise ValueError(
                "The loaded generation config instance is invalid -- `GenerationConfig.validate()` throws warnings "
                "and/or exceptions. Fix these issues to train your model.\n\nThrown during validation:\n" + str(exc)
            )
        return gen_config

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> "PredictionOutput":

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self.model)
        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        summon_full_params_context = (
            FullyShardedDataParallel.summon_full_params(self.model)
            if isinstance(self.model, FullyShardedDataParallel)
            else contextlib.nullcontext()
        )

        with summon_full_params_context:
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.processing_class is not None and hasattr(self.processing_class, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.processing_class.pad_token_id
                if self.processing_class.pad_token_id is not None
                else self.processing_class.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")


        train_dataset_weights, train_dataset_arch =  torch.utils.data.random_split(self.train_dataset, [0.5, 0.5])

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset_weights, datasets.Dataset) and isinstance(train_dataset_arch, datasets.Dataset):
            train_dataset_weights = self._remove_unused_columns(train_dataset_weights, description="training")
            train_dataset_arch = self._remove_unused_columns(train_dataset_arch, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        weights_dataloader_params = dataloader_params.copy()
        arch_dataloader_params = dataloader_params.copy()
        if not isinstance(train_dataset_weights, torch.utils.data.IterableDataset):
            weights_dataloader_params["drop_last"] = self.args.dataloader_drop_last
            weights_dataloader_params["worker_init_fn"] = seed_worker
            weights_dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            weights_dataloader_params["sampler"] = RandomSampler(train_dataset_weights)
        train_dataloader_weights = self.accelerator.prepare(DataLoader(train_dataset_weights, **weights_dataloader_params))

        if not isinstance(train_dataset_arch, torch.utils.data.IterableDataset):
            arch_dataloader_params["drop_last"] = self.args.dataloader_drop_last
            arch_dataloader_params["worker_init_fn"] = seed_worker
            arch_dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            arch_dataloader_params["sampler"] = RandomSampler(train_dataset_arch)
        train_dataloader_arch = self.accelerator.prepare(DataLoader(train_dataset_arch, **arch_dataloader_params))

        return train_dataloader_weights, train_dataloader_arch
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs  
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # self.scaler.scale(loss).backward()
        return loss
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        optimizer = self.optimizer
        optimizer_q = self.optimizer_q
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer, optimizer_q=optimizer_q) # 

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        self.optimizer, self.optimizer_q = self.model.configure_optimizers(self.args.weight_decay, self.args.learning_rate, [self.args.adam_beta1, self.args.adam_beta2]) #
        return self.optimizer, self.optimizer_q
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None, optimizer_q: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        if self.lr_scheduler_q is None:
            self.lr_scheduler_q = get_scheduler(
                'cosine',
                optimizer=self.optimizer_q if optimizer_q is None else optimizer_q,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler_q = True

        return self.lr_scheduler, self.lr_scheduler_q

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, grad_norm_q, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if grad_norm_q is not None:
                logs['grad_norm_q'] = grad_norm_q.detach().item() if isinstance(grad_norm_q, torch.Tensor) else grad_norm_q
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
       
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader_weights, train_dataloader_arch = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader_weights) and has_length(train_dataloader_arch):
            len_dataloader = len(train_dataloader_weights) + len(train_dataloader_arch)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader_weights) + self.num_examples(train_dataloader_arch)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        (self.num_tokens(train_dataloader_weights, args.max_steps) + self.num_tokens(train_dataloader_arch, args.max_steps)) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = (self.num_examples(train_dataloader_weights) + self.num_examples(train_dataloader_arch))* args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = (self.num_tokens(train_dataloader_weights) + self.num_tokens(train_dataloader_arch)) * args.num_train_epochs

        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            import sys
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = (
                        (self.num_tokens(train_dataloader_weights, args.max_steps) + self.num_tokens(train_dataloader_arch, args.max_steps)) * args.gradient_accumulation_steps
                    )
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )
        
        
        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False
        if self._created_lr_scheduler_q:
            self.lr_scheduler_q = None
            self._created_lr_scheduler_q = False

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

       
        self.scaler = torch.cuda.amp.GradScaler()
        


        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}")

        start_time = time.time()
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

      

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        arch_tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self.optimizer.zero_grad()
        self.optimizer_q.zero_grad()
        self.architect.optimizer.zero_grad()

        total_batched_samples = 0
        iter_num = 0
        
        t0 = time.time()
        ctx = (
            torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        )
        time_str = time.strftime("%Y%m%d-%H%M%S")
        out_dir = f"output_tinystories/out_search_{max_steps}_{time_str}"

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        arch_traj_path_file = os.path.join(out_dir, "arch_traj.pkl")
        train_iter_weights = iter(train_dataloader_weights)
        train_iter_arch = iter(train_dataloader_arch)

        batch_sample_arch = next(train_iter_arch)
         
        batch_sample_weights = next(train_iter_weights)

        best_val_loss = 1e9
        weights_loss = -1.0
        local_iter_num = 0
        self.current_flos = 0
        args.max_steps=10000
        args.logging_steps = 500
        args.eval_steps = 10
        annealing_schedule = CosineSched(
            start_step=int(max_steps * args.warmup_ratio),
            max_step=args.max_steps,
            eta_start=0,
            eta_end=0.1
        )
        num_updates = 0
        while True:
            lr = self.get_lr(iter_num, int(max_steps * args.warmup_ratio), args.learning_rate, args.lr_decay_iters, args.min_lr)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            if iter_num % args.logging_steps == 0:
                losses = self.estimate_loss(ctx, train_dataloader_weights, train_dataloader_arch, args.eval_iters)
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        checkpoint = {
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "arch_optimizer": self.architect.optimizer.state_dict(),
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                        }
                        print(f"saving checkpoint to {out_dir}")
                        torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                        arch_config_trajectory = self.model.get_best_config()
                        with open(arch_traj_path_file, "wb") as f:
                            pickle.dump(arch_config_trajectory, f)
                        del arch_config_trajectory

                        print(
                            "New best val loss! Saving arch trajectory to {}".format(
                                arch_traj_path_file
                            )
                        )
                self.model.sampler.before_epoch()
                del losses

            for micro_step in range(args.gradient_accumulation_steps):
                with ctx:
                    arch_output = self.model(**batch_sample_arch)
                    arch_loss = arch_output.loss
                    arch_loss = (
                        arch_loss / args.gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation
                    try:
                        batch_sample_arch = next(train_iter_arch)
                    except StopIteration:
                        train_iter_arch = iter(train_dataloader_arch)
                        batch_sample_arch = next(train_iter_arch)

                self.scaler.scale(arch_loss).backward()
                if iter_num % args.logging_steps == 0:
                    for p in self.model.get_arch_parameters():
                        if p.grad is None:
                            print("Something is wrong")
            if args.max_grad_norm > 0:
                self.scaler.unscale_(self.architect.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.get_arch_parameters(), 5.0)
            self.scaler.step(self.architect.optimizer)
            self.scaler.update()
            self.architect.optimizer.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_q.zero_grad(set_to_none=True)
            
            for micro_step in range(args.gradient_accumulation_steps):
                with ctx:
                    output = self.model(**batch_sample_weights)
                    weights_loss = output.loss
                    # QE_loss, distribution_loss = self.model.auxiliary_quantized_loss(quantization_error_minimization=True)
                    # QE_loss_weight = annealing_schedule(num_updates)

                    overall_loss = weights_loss #+ QE_loss * QE_loss_weight + distribution_loss
                    overall_loss = (
                        overall_loss / args.gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation
                    
                    self.current_flos += float(self.floating_point_ops(batch_sample_weights))
                    try:
                        batch_sample_weights = next(train_iter_weights)
                    except StopIteration:
                        train_iter_weights = iter(train_dataloader_weights)
                        batch_sample_weights = next(train_iter_weights)
                # self.scaler.scale(weights_loss).backward()
                overall_loss.backward()
                
            if args.max_grad_norm > 0:
                # self.scaler.unscale_(self.optimizer)
                # self.scaler.unscale_(self.optimizer_q)
                # for mn, m in self.model.named_modules():
                #     for pn, p in m.named_parameters():
                #         fpn = '%s.%s' % (mn, pn) if mn else pn
                #         if pn.endswith('a_scale') or pn.endswith('w_scale'):
                #             print(pn, p.grad.view(-1))
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.get_model_parameters(), args.max_grad_norm)
                # grad_norm_q = torch.nn.utils.clip_grad_norm_(self.model.get_quantization_parameters(), args.max_grad_norm)
                
                # grad_norm_q = torch.nn.utils.clip_grad_norm_(self.model.get_quantization_parameters(), args.max_grad_norm)
                # grad_norm_q = torch.nn.utils.clip_grad_norm_(self.model.get_quantization_parameters(), args.max_grad_norm)
            # self.scaler.step()
            self.optimizer.step()
            # self.optimizer_q.step()
            # self.scaler.step(self.optimizer_q)
            # self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            # self.optimizer_q.zero_grad(set_to_none=True)
            self.architect.optimizer.zero_grad(set_to_none=True)
            num_updates += 1
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            weights_lossf = weights_loss.item() * args.gradient_accumulation_steps
            # QE_lossf = QE_loss.item() * args.gradient_accumulation_steps
            overall_lossf = overall_loss.item() * args.gradient_accumulation_steps
            # print(
            #     f"iter {iter_num}: loss {weights_lossf:.4f}, QE loss {QE_lossf:.4f}, Overall loss {overall_lossf:.4f}, grad_norm {grad_norm:.2f}, grad_norm_q {grad_norm_q:.2f} "
            # )
            print(
                f"iter {iter_num}: loss {weights_lossf:.4f}, grad_norm {grad_norm:.2f}"
            )
            
                # for k in self.model.arch_parameter_dict.keys():
                #     print(
                #         f"arch parameter {k}: {torch.nn.functional
                # .softmax(self.model.arch_parameter_dict[k], dim=-1)}"
                #     )
                
            if iter_num % args.logging_steps == 0:
                print(f"Best config: {self.model.get_best_config()}")
                
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > args.max_steps:
                break
            if math.isnan(grad_norm) : # or math.isnan(grad_norm_q)
                break
        metrics = {}
        metrics["total_flos"] = self.current_flos
        metrics["train_loss"] = overall_lossf
        print(args.max_steps, iter_num, "test")

        return TrainOutput(iter_num, overall_lossf, metrics)
    
    def estimate_loss(self, ctx, train_dataloader_weights, train_dataloader_arch, eval_iters):
        out = {}
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for i, data in enumerate(iter(train_dataloader_weights)):
            if i >= eval_iters:
                break
            with ctx:
                output = self.model(**data)
            losses[i] = output.loss.detach()
        out["train"] = losses.mean()

        losses = torch.zeros(eval_iters)
        for i, data in enumerate(iter(train_dataloader_arch)):
            if i >= eval_iters:
                break
            with ctx:
                output = self.model(**data)
            losses[i] = output.loss.detach()
        out["val"] = losses.mean()

        self.model.train()
        return out

    def get_lr(self, it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    
    