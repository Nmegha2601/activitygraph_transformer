#!/bin/bash

DATASET=$1
NCLASSES=$2
SR=$3
NSTEPS=$4
EPOCHS=$5
BATCHSIZE=$6
LR=$7
POSEMB=$8
LRJOINER=$9
SAVEFREQ=${10}
NQUERIES=${11}
NENCLAYERS=${12}
NDECLAYERS=${13}
HDIM=${14}
NHEADS=${15}
NPOSEMB=${16}
DROPOUT=${17}
LRDROP=${18}
WDECAY=${19}
CLIPNORM=${20}
NF=${21}

OUT="output/checkpoints_"${FOLDER_SUFFIX}"/checkpoints_numqueries"${NQUERIES}"_lr"$LR"_lrdrop"${LRDROP}"_dropout"${DROPOUT}"_clipmaxnorm"${CLIPNORM}"_weightdecay"${WDECAY}"_posemb"$POSEMB"_lrjoiner"${LRJOINER}"_nheads"${NHEADS}"_nenclayers"${NENCLAYERS}"_ndeclayers"${NDECLAYERS}"_hdim"${HDIM}"_sr"${SR}"_batchsize"${BATCHSIZE}"_nposembdict"${NPOSEMB}"_numinputs"${NF}

LOGDIR="output/logs"
LOG=${LOGDIR}"/log_numqueries"${NQUERIES}"_lr"$LR"_lrdrop"${LRDROP}"_dropout"${DROPOUT}"_clipmaxnorm"${CLIPNORM}"_weightdecay"${WDECAY}"_posemb"$POSEMB"_lrjoiner"${LRJOINER}"_nheads"${NHEADS}"_nenclayers"${NENCLAYERS}"_ndeclayers"${NDECLAYERS}"_hdim"${HDIM}"_sr"${SR}"_batchsize"${BATCHSIZE}"_nposembdict"${NPOSEMB}"_numinputs"${NF}".log"

mkdir -p ${LOGDIR}
exec &> >(tee -a "${LOG}")
echo Logging output to "${LOG}"

DATA=$PWD"/data/"${DATASET}

python src/main.py --dataset ${DATASET} --data_root ${DATA} --model "agt" --features "i3d_feats" --batch_size ${BATCHSIZE} --enc_layers ${NENCLAYERS} --dec_layers ${NDECLAYERS} --num_queries ${NQUERIES} --nheads ${NHEADS} --dropout ${DROPOUT} --weight_decay ${WDECAY} --clip_max_norm ${CLIPNORM} --num_inputs ${NF} --num_pos_embed_dict ${NPOSEMB} --hidden_dim ${HDIM} --num_workers 0 --num_classes ${NCLASSES} --step_size ${NSTEPS} --sample_rate ${SR}  --lr ${LR} --lr_drop ${LRDROP} --epochs ${EPOCHS} --output_dir ${OUT} --position_embedding ${POSEMB} --lr_joiner ${LRJOINER} --save_checkpoint_every ${SAVEFREQ} --dim_feedforward $(( 4 * ${HDIM} )) --set_cost_segment 5 --set_cost_siou 3 --segment_loss_coef 5 --siou_loss_coef 3 


