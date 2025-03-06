# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from coconut import Coconut
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
from copy import copy
import os, sys
import yaml
import json
import gc
import argparse
from utils import Config, set_seed


def evaluate_model(model, dataloader, tokenizer, answers_val, cot_val, question_val, max_new_tokens, 
                  distributed=False, rank=0, world_size=1, wandb_run=None):
    """
    Evaluate the model on the validation set and return accuracy metrics.
    
    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        tokenizer: Tokenizer for decoding outputs
        answers_val: List of ground truth answers
        cot_val: List of ground truth chain-of-thought reasoning
        question_val: List of questions
        max_new_tokens: Maximum number of tokens to generate
        distributed: Whether distributed evaluation is being used
        rank: Current process rank
        world_size: Total number of processes
        wandb_run: Wandb run object for logging
        
    Returns:
        tuple: (accuracy, cot_accuracy)
    """
    total_length = len(dataloader)
    pbar = tqdm(
        colour="blue", desc="Evaluating", total=total_length, dynamic_ncols=True
    )
    
    cor, cor_cot, total = (
        torch.tensor(0, device=rank if distributed else 0),
        torch.tensor(0, device=rank if distributed else 0),
        torch.tensor(0, device=rank if distributed else 0),
    )

    with torch.no_grad():
        model_to_eval = model.module if distributed else model
        model_to_eval.eval()
        
        for idx, batch in enumerate(dataloader):
            test_idx = batch["idx"][0]

            batch = {
                k: v.to(rank if distributed else 0)
                for k, v in batch.items()
                if v is not None and k not in ["idx", "position_ids"]
            }

            assert len(batch["input_ids"]) == 1
            answer = answers_val[test_idx.cpu().item()]
            answer_cot = cot_val[test_idx.cpu().item()]
            question = question_val[test_idx.cpu().item()]

            total += 1

            outputs = model_to_eval.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                synced_gpus=distributed,
            )

            text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_output = text_output.split("#")[-1].replace(",", "").strip()
            cot_output = (
                ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
            )

            if idx < 5 and rank == 0:
                # print some examples
                print(
                    f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'"
                )
                print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                print(f"Extracted Output: '{answer_output}'")

            cor += answer_output == answer
            cor_cot += cot_output == answer_cot

            pbar.update(1)
            pbar.set_description(
                f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
            )

        pbar.close()
        print(f"Device {rank}: Cor={cor}, CoT={cor_cot}, Total={total}")

    # Only reduce metrics in distributed mode
    if distributed:
        dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    cor_cot = cor_cot.item()
    cor = cor.item()
    total = total.item()
    
    if rank == 0:
        print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
        print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        
        if wandb_run:
            wandb_run.log({"eval/acc": cor / total, "eval/cot_em": cor_cot / total})
    
    return cor/total, cor_cot/total


def evaluate_loss(model, dataloader, distributed=False, rank=0, world_size=1, wandb_run=None):
    """
    Evaluate the model's loss on the validation set.
    
    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        distributed: Whether distributed evaluation is being used
        rank: Current process rank
        world_size: Total number of processes
        wandb_run: Wandb run object for logging
        
    Returns:
        float: Average loss
    """
    total_loss = 0

    with torch.no_grad():
        model_to_eval = model.module if distributed else model
        model_to_eval.eval()
        
        for step, batch in enumerate(dataloader):
            batch = {
                key: batch[key].to(rank if distributed else 0) 
                for key in batch.keys() if key != "idx"
            }

            outputs = model(**batch)
            loss = outputs.loss
            
            if distributed:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                total_loss += loss.item() / world_size
            else:
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        
        if rank == 0:
            print(f"Validation loss: {avg_loss}")
            if wandb_run:
                wandb_run.log({"eval/loss": avg_loss})
                
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="train router only in coconut")
    parser.add_argument("config_file", help="Path to the YAML configuration file")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only without training")
    args = parser.parse_args()

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    
    # Override config with command line argument if specified
    if args.eval_only:
        config_dict["only_eval"] = True

    configs = Config(config_dict)

    # Initialize variables for distributed training
    local_rank = 0
    rank = 0
    world_size = 1

    # init distributed environment
    if config_dict["distributed_training"]:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
    else:
        # For non-distributed training, use device 0
        torch.cuda.set_device(0)

    if rank == 0:
        print("Config:", config_dict)
        if config_dict["only_eval"]:
            print("Running in evaluation-only mode")

    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    # Only use barrier in distributed mode
    if config_dict["distributed_training"]:
        torch.distributed.barrier()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Create Coconut model
    model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    
    # Load the trained Coconut checkpoint
    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path, map_location=torch.device(rank)
        )
        model.load_state_dict(saved_weights, strict=False)
        if rank == 0:
            print(f"Loaded checkpoint from {configs.load_model_path}")
    else:
        raise ValueError("Must provide a Coconut checkpoint to train router only")

    # Freeze parameters if necessary
    for name, param in model.named_parameters():
        if config_dict["train_router_only"]:
            # Only train router parameters
            param.requires_grad = "router" in name
            if rank == 0 and param.requires_grad:
                print(f"Training parameter: {name}")
        elif config_dict["train_base_model_only"]:
            # Only train base model parameters
            param.requires_grad = "router" not in name
            if rank == 0 and param.requires_grad:
                print(f"Training parameter: {name}")
        else:
            param.requires_grad = False
    
    model = model.to(rank if config_dict["distributed_training"] else 0)
    
    # Wrap model in DDP only if distributed training is enabled
    if config_dict["distributed_training"]:
        parallel_model = DDP(model, device_ids=[rank])
    else:
        parallel_model = model  # Use the model directly for non-distributed training

    if rank == 0:
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params} / {total_params} ({trainable_params/total_params:.2%})")

    # prepare the ground truth answer and cot for evaluation
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )
    base_dataset_train = get_dataset(
        configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000
    )

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0

    if not configs.debug and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])
    else:
        wandb_run = None

    # Use a smaller learning rate for fine-tuning just the router
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, parallel_model.parameters()),
        lr=configs.lr * 0.1,  # Lower learning rate for fine-tuning
        weight_decay=configs.weight_decay,
    )

    best_acc = 0
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    # Skip training loop if in eval-only mode
    if not configs.only_eval:
        for epoch in range(configs.resume, configs.num_epochs):
            # Always use the maximum latent stage since we're fine-tuning the router
            scheduled_stage = configs.max_latent_stage
            
            dataset_gen_val = get_question_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=False,
            )

            valid_gen_dataloader = torch.utils.data.DataLoader(
                dataset_gen_val,
                num_workers=1,
                pin_memory=True,
                batch_size=1,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_gen_val, shuffle=False) if config_dict["distributed_training"] else None
            )

            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=False,
                shuffle=True,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True) if config_dict["distributed_training"] else None,
            )

            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=False,
            )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False) if config_dict["distributed_training"] else None,
            )

            parallel_model.module.train() if config_dict["distributed_training"] else parallel_model.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):
                if step == 0 and wandb_run and rank == 0:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    wandb_run.log({"data_table": copy(text_table)})

                total_train_steps += 1
                batch = {
                    key: batch[key].to(rank if config_dict["distributed_training"] else 0) for key in batch.keys() if key != "idx"
                }

                outputs = parallel_model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            dist.barrier()

            if not configs.debug and rank == 0:
                states = parallel_model.state_dict()
                torch.save(
                    states, os.path.join(save_dir, f"router_only_checkpoint_{epoch + 1}")
                )
                print("saving model.")

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

            # Evaluate validation loss
            val_loss = evaluate_loss(
                parallel_model, 
                valid_loss_dataloader,
                distributed=config_dict["distributed_training"],
                rank=rank,
                world_size=world_size,
                wandb_run=wandb_run if not configs.debug and rank == 0 else None
            )
            
            # Evaluate generation accuracy
            accuracy, cot_accuracy = evaluate_model(
                parallel_model,
                valid_gen_dataloader,
                tokenizer,
                answers_val,
                cot_val,
                question_val,
                max_new_tokens,
                distributed=config_dict["distributed_training"],
                rank=rank,
                world_size=world_size,
                wandb_run=wandb_run if not configs.debug and rank == 0 else None
            )
            
            # Only use barrier in distributed mode
            if config_dict["distributed_training"]:
                dist.barrier()
            
            # Save model
            if accuracy > best_acc and not configs.debug:
                best_acc = accuracy
                if config_dict["distributed_training"]:
                    states = parallel_model.module.state_dict()
                else:
                    states = parallel_model.state_dict()

                if rank == 0:
                    torch.save(states, os.path.join(save_dir, f"router_only_best_checkpoint"))
                    print("saving best model.")

                if config_dict["distributed_training"]:
                    dist.barrier()
            
                del states
                gc.collect()
                torch.cuda.empty_cache()

    else:
        if rank == 0:
            print("Skipping training, running evaluation only...")


if __name__ == "__main__":
    main() 