# launch launch.sh --classical_logfile_names --gpu_type rtxa5000 --mem 128 --timelimit 24 --gpus 1

python eval.py --model /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v2 \
    --dataset DynaGuard/promptpatrol --subset combined --split val --sample_size 30

python eval.py --model /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v3 \
    --dataset DynaGuard/promptpatrol --subset combined --split val --sample_size 30

python eval.py --model /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v4 \
    --dataset DynaGuard/promptpatrol --subset combined --split val --sample_size 30

python eval.py --model /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v5 \
    --dataset DynaGuard/promptpatrol --subset combined --split val --sample_size 30

python eval.py --model /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v6 \
    --dataset DynaGuard/promptpatrol --subset combined --split val --sample_size 30

python eval.py --model /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v7 \
    --dataset DynaGuard/promptpatrol --subset combined --split val --sample_size 30

python eval.py --model /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v8 \
    --dataset DynaGuard/promptpatrol --subset combined --split val --sample_size 30

python eval.py --model /fs/cml-projects/guardian_models/verl/Andrew_runs2/eval/v9 \
    --dataset DynaGuard/promptpatrol --subset combined --split val --sample_size 30

#### Evaluate Dynaguard
python eval.py --model DynaGuard/DynaGuard-8B-6750 \
    --dataset DynaGuard/promptpatrol --subset combined --split val --sample_size 30