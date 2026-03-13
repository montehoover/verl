# launch launch-unshard.sh --classical_logfile_names --gpu_type rtxa5000 --mem 128 --timelimit 24 --gpus 1

# python scripts/legacy_model_merger.py merge \
#     --backend fsdp \
#     --local_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/DynaGuard_DynaGuard-8B-6750_code_patrol_postprocessed_lr1e-6_bs4_subsetv2/global_step_75/actor \
#     --target_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v2

# python scripts/legacy_model_merger.py merge \
#     --backend fsdp \
#     --local_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/DynaGuard_DynaGuard-8B-6750_code_patrol_postprocessed_lr1e-6_bs4_subsetv3/global_step_75/actor \
#     --target_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v3

# python scripts/legacy_model_merger.py merge \
#     --backend fsdp \
#     --local_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/DynaGuard_DynaGuard-8B-6750_code_patrol_postprocessed_lr1e-6_bs4_subsetv4/global_step_75/actor \
#     --target_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v4

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/DynaGuard_DynaGuard-8B-6750_code_patrol_postprocessed_lr1e-6_bs4_subsetv5/global_step_74/actor \
    --target_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v5

# python scripts/legacy_model_merger.py merge \
#     --backend fsdp \
#     --local_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/DynaGuard_DynaGuard-8B-6750_code_patrol_postprocessed_lr1e-6_bs4_subsetv6/global_step_75/actor \
#     --target_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v6

# python scripts/legacy_model_merger.py merge \
#     --backend fsdp \
#     --local_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/DynaGuard_DynaGuard-8B-6750_code_patrol_postprocessed_lr1e-6_bs4_subsetv7/global_step_75/actor \
#     --target_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v7

# python scripts/legacy_model_merger.py merge \
#     --backend fsdp \
#     --local_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/DynaGuard_DynaGuard-8B-6750_code_patrol_postprocessed_lr1e-6_bs4_subsetv8/global_step_75/actor \
#     --target_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v8

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/DynaGuard_DynaGuard-8B-6750_code_patrol_postprocessed_lr1e-6_bs4_subsetv9/global_step_71/actor \
    --target_dir /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v9