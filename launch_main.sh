# Run with:
# launch launch_main.sh --classical_logfile_names --gpu_type rtxa5000 --mem 30

python main.py --model Qwen/Qwen3-8B --dataset tomg-group-umd/compliance --split train_80k_mix --run_sft --sft_lr 1e-5 --sft_lr_schedule cosine --run_grpo --grpo_lr 1e-6 --grpo_examples 9000 --checkpoint_dir /fs/cml-projects/guardian_models/verl --resume_grpo
python main.py --model Qwen/Qwen3-8B --dataset tomg-group-umd/compliance --split train_80k_mix --num_examples 9000 --run_grpo --grpo_lr 1e-6 --checkpoint_dir /fs/cml-projects/guardian_models/verl