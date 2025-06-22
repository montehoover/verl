import multiprocessing
import argparse, os, subprocess, sys
import torch
from verl.compliance.helpers import check_pytorch_cuda_error, configure_logging, convert_to_hf, get_auc, get_last_checkpoint_path, get_model_name, get_subset, prepare_dataset_for_verl, push_to_hf_hub, run_subprocess
import wandb


def main(args):

    os.environ["WANDB_ENTITY"] = args.wandb_entity
    model_path = args.model
    num_examples = args.num_examples
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs found. Double check the that this is being called where you expect it to be."
    train_files, val_files, num_train_examples = prepare_dataset_for_verl(
        dataset_path=args.dataset,
        subset=args.subset,
        split=args.split,
        num_examples=num_examples,
        num_val_examples=args.num_val_examples,
        redownload=args.redownload,
        local_dir="data/compliance",
        model_path=model_path,
        do_rules_rewards=args.do_rules_rewards,
        val_dataset_split=args.val_split,
        cuda_mem_test_len=args.cuda_mem_test_len,
    )

    ################################
    # SFT Training
    ################################
    if args.run_sft:
        print("\nStarting SFT...\n")
        model_name = get_model_name(model_path)
        sft_run_name = f"{model_name}_{args.split}_sft_lr{args.sft_lr}_bs{args.sft_batch_size}_ep{args.sft_epochs}_{args.sft_lr_schedule[:3]}"

        # Alway test the same numbe of times per epoch, regardless of batch size, so that each run has validation results after the same number of examples
        # batch 256, 16k examples: test at  6, 12...
        # batch 128, 16k examples: test at 12, 24...
        # batch 64, 16k examples: test at  24, 48...
        if args.val_steps_per_epoch == 0:
            val_freq = -1
        else:
            val_freq = int(num_train_examples / (args.sft_batch_size * args.val_steps_per_epoch))

        if args.resume_grpo:
            # No need to rerun SFT. But we want to be inside the sft section if --run_sft was given so we pick up the correct sft_run_name for the GRPO checkpoint.
            pass
        else:
            sft_cmd = [
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                f"-m", "verl.trainer.fsdp_sft_trainer",
                f"model.partial_pretrain={model_path}",
                f"model.enable_gradient_checkpointing=True",
                f"data.train_files={train_files}",
                f"data.val_files={val_files}",
                f"data.prompt_key=extra_info",
                f"data.response_key=extra_info",
                f"data.prompt_dict_keys=['question']",
                f"data.response_dict_keys=['answer']",
                f"+data.add_system_prompt=True",
                f"data.max_length={args.max_prompt_length}",
                f"data.filter_overlong_prompts={args.filter_long_prompts}",
                f"data.truncation=left",
                f"data.micro_batch_size_per_gpu={args.sft_batch_size_per_gpu}",
                f"data.train_batch_size={args.sft_batch_size}",
                f"optim.lr={args.sft_lr}",
                f"optim.lr_scheduler={args.sft_lr_schedule}",
                f"trainer.total_epochs={args.sft_epochs}",
                f"trainer.logger=['console','wandb']",
                f"trainer.project_name={args.sft_wandb_project}",
                f"trainer.experiment_name={sft_run_name}",
                f"trainer.default_local_dir={args.checkpoint_dir}/{sft_run_name}",
                f"trainer.test_freq={val_freq}",
            ]
            subprocess.run(sft_cmd, check=True)
            last_checkpoint_path = get_last_checkpoint_path(sft_run_name, checkpoint_dir=args.checkpoint_dir)
            print(f"\nSuccessfully completed SFT. Last checkpoint was saved to {last_checkpoint_path}\n")
            
            # Push to Hugging Face Hub and use that model_path for GRPO
            model_path = push_to_hf_hub(checkpoint_path=last_checkpoint_path, run_name=sft_run_name, original_model=model_path)

    ################################
    # GRPO Training
    ################################
    if args.run_grpo:
        print("\nStarting GRPO...\n")
        if args.grpo_examples != -1:
            num_examples = args.grpo_examples
            train_files = get_subset(train_files, num_examples)
        
        model_name = get_model_name(model_path)
        grpo_run_name = f"{model_name}_{args.split}_grpo_ex{num_examples}_lr{args.grpo_lr}_bs{args.grpo_batch_size}_len{args.max_response_length}"
        
        if args.run_sft:
            grpo_run_name = grpo_run_name.replace(f"{model_name}_{args.split}", sft_run_name)
        if args.resume_grpo:
            resume_mode = "auto"
        else:
            resume_mode = "disable"
        grpo_cmd = [
            "python3",   
            "-m", 
            "verl.trainer.main_ppo",
            f"algorithm.adv_estimator=grpo",
            f"data.train_files={train_files}",
            f"data.val_files={val_files}",
            f"data.train_batch_size={args.grpo_batch_size}",
            f"data.max_prompt_length={args.max_prompt_length}",
            f"data.max_response_length={args.max_response_length}",
            f"data.filter_overlong_prompts=True",
            f"data.truncation=error",
            f"actor_rollout_ref.model.path={model_path}",
            f"actor_rollout_ref.actor.optim.lr={args.grpo_lr}",
            f"actor_rollout_ref.actor.optim.warmup_style={args.grpo_lr_schedule}",
            f"actor_rollout_ref.model.use_remove_padding=True",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={args.num_generations}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.grpo_batch_size_per_gpu}",
            f"actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef=0.001",
            f"actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            f"actor_rollout_ref.actor.entropy_coeff=0",
            f"actor_rollout_ref.model.enable_gradient_checkpointing=True",
            f"actor_rollout_ref.actor.fsdp_config.param_offload=False",
            f"actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.grpo_batch_size_per_gpu}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size=1", # Verl uses 2 when doing a 7B model on 8 GPUs. They use 4 when doing a 32B model on 32 GPUs. 
            f"actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={args.vllm_cache_utilization}", # Use 0.7 for 14B models, 0.6 for all smaller models
            f"actor_rollout_ref.rollout.n={args.num_generations}",
            f"actor_rollout_ref.rollout.enable_chunked_prefill=False",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.grpo_batch_size_per_gpu}",
            f"actor_rollout_ref.ref.fsdp_config.param_offload=True",
            f"actor_rollout_ref.rollout.enable_chunked_prefill=False",
            f"algorithm.use_kl_in_reward=False",
            f"custom_reward_function.path=verl/compliance/helpers.py",
            f"custom_reward_function.name=compute_reward",
            f"trainer.critic_warmup=0",
            f"trainer.logger=['console','wandb']",
            f"trainer.project_name={args.grpo_wandb_project}",
            f"trainer.experiment_name={grpo_run_name}",
            f"trainer.n_gpus_per_node={num_gpus}",
            f"trainer.nnodes=1",
            f"trainer.save_freq=20",
            f"trainer.test_freq=-1",
            f"trainer.max_actor_ckpt_to_keep=1",
            f"trainer.resume_mode={resume_mode}",
            f"trainer.total_epochs={args.grpo_epochs}",
            f"trainer.default_local_dir={args.checkpoint_dir}/{grpo_run_name}",
        ]
        # Override Verl's initization of wandb so we can log AUC after training
        if args.is_hp_sweep:
            wandb.init()
        else:
            wandb.init(project=args.grpo_wandb_project, name=grpo_run_name)
        
        subprocess.run(grpo_cmd, check=True)

        last_checkpoint_path = get_last_checkpoint_path(grpo_run_name, checkpoint_dir=args.checkpoint_dir)
        print(f"\nSuccessfully completed GRPO. Last checkpoint was saved to {last_checkpoint_path}\n")
        
        hf_format_model_path = convert_to_hf(last_checkpoint_path, model_path)
        auc = get_auc(hf_format_model_path, val_files)
        wandb.log({"val-core/auc": auc})
        wandb.summary["val-core/auc"] = auc # show in sweep summary table

        # Push to Hugging Face Hub
        model_path = push_to_hf_hub(checkpoint_path=last_checkpoint_path, run_name=grpo_run_name, original_model=model_path, delete_checkpoint=args.delete_checkpoint_on_push)
        wandb.finish()

    print("Training process completed successfully.")


    

def parse_args():
    parser = argparse.ArgumentParser(description="This script runs a PPO training process with configurable parameters.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name (default: Qwen/Qwen3-0.6B)")
    
    # Dataset
    parser.add_argument("--dataset", default="tomg-group-umd/compliance", help="Dataset name (default: tomg-group-umd/compliance)")
    parser.add_argument("--subset", default="compliance", help="Dataset subset (default: compliance)")
    parser.add_argument("--split", default="train_32000_mix", help="Dataset split (default: train_cot)")
    parser.add_argument("--val_split", default=None, help="Validation dataset split (default: val_256)")
    parser.add_argument("--num_examples", type=int, default=-1, help="Number of examples to train on. -1 for all (default: -1)")
    parser.add_argument("--num_val_examples", type=int, default=256, help="Number of examples for validation (default: 256)")
    parser.add_argument("--redownload", default=True, action=argparse.BooleanOptionalAction, help="Force redownload of data (default: disabled)")
    parser.add_argument("--max_prompt_length", default=8192, type=int, help="Max prompt length (default: 8192)")
    parser.add_argument("--vllm_cache_utilization", default=0.6, type=float, help="VLLM cache utilization (default: 0.6). Set to 0.7 for 14B models, 0.6 for all smaller models.")
    
    # Run info
    parser.add_argument("--run_sft", default=False, action=argparse.BooleanOptionalAction, help="Run SFT (default: enabled)")
    parser.add_argument("--run_grpo", default=False, action=argparse.BooleanOptionalAction, help="Run GRPO (default: enabled)")
    parser.add_argument("--download_local_dir", default="data/compliance", help="Local directory for data (default: data/compliance)")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory for checkpoints (default: checkpoints)")
    parser.add_argument("--exit_on_checkpoint_error", default=True, action=argparse.BooleanOptionalAction, help="Exit on checkpoint error (default: enabled)")
    parser.add_argument("--wandb_entity", default="guardian-models", help="Weights & Biases entity (default: guardian-models)")
    parser.add_argument("--is_hp_sweep", default=False, action=argparse.BooleanOptionalAction, help="Is this a hyperparameter sweep? (default: disabled)")
    parser.add_argument("--delete_checkpoint_on_push", default=True, action=argparse.BooleanOptionalAction, help="Delete checkpoint after pushing to Hugging Face Hub (default: enabled)")
    parser.add_argument("--cuda_mem_test_len", default=None, type=int, help="Test CUDA memory with this many tokens (default: disabled)")
    parser.add_argument("--val_steps_per_epoch", default=10, type=int, help="Frequency of testing during training (0 for never)")

    # SFT
    parser.add_argument("--sft_wandb_project", default="sft-compliance", help="Trainer project name for WandB (default: sft-compliance)")
    parser.add_argument("--sft_epochs", default=1, type=int, help="Number of epochs (default: 4)")
    parser.add_argument("--sft_lr", default="1e-5", help="Learning rate (default: 1e-5)")
    parser.add_argument("--sft_batch_size", default=128, type=int, help="Total batch size (default: 128)")
    parser.add_argument("--sft_batch_size_per_gpu",default=1, type=int, help="Batch size per GPU. Reduce if hitting OOM. Increase for faster training. (default: 2)")
    parser.add_argument("--sft_lr_schedule", default="cosine", help="Learning rate schedule (default: cosine)", choices=["cosine", "constant"])
    parser.add_argument("--filter_long_prompts", default=True, action=argparse.BooleanOptionalAction, help="Filter overlong prompts (default: disabled)")

    # GRPO
    parser.add_argument("--grpo_wandb_project", default="grpo-compliance", help="Trainer project name for WandB (default: grpo-compliance)")
    parser.add_argument("--grpo_epochs", default=1, type=int, help="Number of epochs (default: 1)")
    parser.add_argument("--grpo_examples", default=-1, type=int, help="Number of examples to do GRPO on if different from SFT (default: -1, same as SFT)")
    parser.add_argument("--grpo_lr", default="1e-6", help="Learning rate (default: 1e-6)")
    parser.add_argument("--grpo_batch_size", default=64, type=int, help="Total batch size (default: 48)")
    parser.add_argument("--grpo_batch_size_per_gpu",default=2, type=int, help="Batch size per GPU. Reduce if hitting OOM. Increase for faster training. (default: 2)")
    parser.add_argument("--num_generations", default=8, type=int, help="Number of generations (default: 12)")
    parser.add_argument("--max_response_length", default=1024, type=int, help="Max response length (default: 1024)")
    parser.add_argument("--grpo_lr_schedule", default="cosine", help="Learning rate schedule (default: cosine)", choices=["cosine", "constant"])
    parser.add_argument("--resume_grpo", default=False, action=argparse.BooleanOptionalAction, help="Resume GRPO training from last checkpoint (default: disabled)")
    parser.add_argument("--do_rules_rewards", default=False, action=argparse.BooleanOptionalAction, help="Use rules rewards (default: disabled)")
    return parser.parse_args()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()
    main(args)