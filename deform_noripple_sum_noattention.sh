#!/bin/bash
#SBATCH --job-name=deform_noripple_sum_noattention
#SBATCH --partition=gpu_8

i=3
srun --exclusive -N1 -p gpu_8 --gres=gpu python run_model.py --model=deform --mode=all --rollout_split=valid --dataset=deforming_plate  --epochs=25 --trajectories=1000 --num_rollouts=100 --core_model=encode_process_decode --message_passing_aggregator=sum --message_passing_steps=${i} --attention=False --ripple_used=False --ripple_generation=equal_size --ripple_generation_number=1 --ripple_node_selection=random --ripple_node_selection_random_top_n=1 --ripple_node_connection=most_influential --ripple_node_ncross=1 --use_prev_config=True
