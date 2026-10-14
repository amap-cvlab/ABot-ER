# Inspired by: https://github.com/lvwerra/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py

from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorWithPadding

from ...data import get_dataset
from ...extras.callbacks import SaveProcessorCallback
from ...extras.ploting import plot_loss
from ...model import load_model, load_template
from ...train.ppo.trainer import CustomPPOTrainer
from ...train.utils import create_ref_model, create_reward_model


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_ppo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    template = load_template(model_args, template=data_args.template, enable_thinking=data_args.enable_thinking)
    tokenizer, processor = template.tokenizer, template.processor
    dataset_module = get_dataset(tokenizer, model_args, data_args, training_args, stage="ppo", template=template)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=training_args.do_train, add_valuehead=True)

    tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    callbacks = [SaveProcessorCallback(processor)] + callbacks

    # Create reference model and reward model
    ref_model = create_ref_model(model_args, finetuning_args, data_args, add_valuehead=True)
    reward_model = create_reward_model(model, model_args, finetuning_args)

    # Initialize our Trainer
    ppo_trainer = CustomPPOTrainer(
        model_args=model_args,
        training_args=training_args,
        data_args=data_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks,
        reward_model=reward_model,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **dataset_module,
    )

    # Training
    if training_args.do_train:
        ppo_trainer.ppo_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        ppo_trainer.save_model()

        ppo_trainer.save_state()  # must be called after save_model to have a folder
        if ppo_trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward"])
