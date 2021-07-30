# Experiment parameters (modify these to run different experiments)

V_task1='abc' # vocabulary used only in task1 when training
V_task2='xyz' # vocabulary only used in task2 when training
V_both='' # vocabulary used in both tasks when training

# sequence lengths
min_len=1 # shortest sequence
max_len=3 # longest sequences

# version 1: always repeat V_task1, never repeat V_task2 (in train; switch in test)
# version 2: only have V_task1 in repeated datapoints, only have V_task2 in non-repeated datapoints (in train; switch in test)
VERSION=1

MODEL='bert'
EPOCHS=5
EVAL_SPLIT=0.2
USE_CUDA=false # GPU/CPU

LM_BATCH_SIZE=16
CLF_BATCH_SIZE=16

data_folder='data'
lm_folder='lm'
clf_folder='clf'

results='results/results_repetition.txt'
mkdir -p 'results'

echo "$V_task1 $V_both [$min_len, $max_len]" >> $results
echo "$V_task2 $V_both [$min_len, $max_len]" >> $results
echo "Model: $MODEL" >> $results

########################################################################################################################################################################

# Create datasets

if [ $VERSION == 1 ]; then
  python create_datasets.py -task repeat -voc "$V_task1$V_both" --min_len "$min_len" --max_len "$max_len" --repeat "$V_task1" --save_folder "$data_folder" --save_suffix "repeat" --eval_split $EVAL_SPLIT
  python create_datasets.py -task repeat -voc "$V_task1" --min_len "$min_len" --max_len "$max_len" --dont_repeat "$V_task1" --save_folder "$data_folder" --save_suffix "no_repeat" --eval_split 0
  python create_datasets.py -task repeat -voc "$V_task2$V_both" --min_len "$min_len" --max_len "$max_len" --dont_repeat "$V_task2" --save_folder "$data_folder" --save_suffix "no_repeat" --eval_split $EVAL_SPLIT
  python create_datasets.py -task repeat -voc "$V_task2" --min_len "$min_len" --max_len "$max_len" --repeat "$V_task2" --save_folder "$data_folder" --save_suffix "repeat" --eval_split 0
elif [ $VERSION == 2 ]; then
  python create_datasets.py -task repeat -voc "$V_task1$V_both" --min_len "$min_len" --max_len "$max_len" --repeat "$V_task1$V_both" --save_folder "$data_folder" --save_suffix "repeat" --eval_split $EVAL_SPLIT
  python create_datasets.py -task repeat -voc "$V_task1" --min_len "$min_len" --max_len "$max_len" --dont_repeat "$V_task1" --save_folder "$data_folder" --save_suffix "no_repeat" --eval_split 0
  python create_datasets.py -task repeat -voc "$V_task2$V_both" --min_len "$min_len" --max_len "$max_len" --dont_repeat "$V_task2$V_both" --save_folder "$data_folder" --save_suffix "no_repeat" --eval_split $EVAL_SPLIT
  python create_datasets.py -task repeat -voc "$V_task2" --min_len "$min_len" --max_len "$max_len" --repeat "$V_task2" --save_folder "$data_folder" --save_suffix "repeat" --eval_split 0
fi

# Train and test datasets
train_task1="$data_folder/repeat/${V_task1}${V_both}_repeat/train.txt"
eval_task1="$data_folder/repeat/${V_task1}${V_both}_repeat/eval.txt"
test_task1="$data_folder/repeat/${V_task2}_repeat/all.txt"
train_task2="$data_folder/repeat/${V_task2}${V_both}_no_repeat/train.txt"
eval_task2="$data_folder/repeat/${V_task2}${V_both}_no_repeat/eval.txt"
test_task2="$data_folder/repeat/${V_task1}_no_repeat/all.txt"

# Count number of training and eval samples
#printf "Number of task1 training samples: " >> $results
#wc -l $train_task1 >> $results
#printf "Number of task1 evaluation samples: " >> $results
#wc -l $eval_task1 >> $results
#printf "Number of task2 training samples: " >> $results
#wc -l $train_task2 >> $results
#printf "Number of task2 evaluation samples: " >> $results
#wc -l $eval_task2 >> $results
#printf "\n" >> $results

cuda=''
if $USE_CUDA
then
  cuda='--use_cuda'
fi

# Train language model on training data
printf "\nTraining language model\n"
python simpletransformers/train_lm.py --data "$train_task1" "$train_task2" -m "$MODEL" --batch_size $LM_BATCH_SIZE --epochs "$EPOCHS" $cuda

lm="$lm_folder/$MODEL"

# Train classifier on training data
printf "\nTraining classifier\n"
python simpletransformers/train_clf.py --train_data "$train_task1" "$train_task2" --eval_data "$eval_task1" "$eval_task2" -m "$MODEL" -lm $lm --epochs "$EPOCHS" --batch_size $CLF_BATCH_SIZE $cuda

clf="$clf_folder/$MODEL"


# Test trained classifier

printf "Evaluation results: " >> $results
python simpletransformers/test_clf.py -d "$eval_task1" "$eval_task2" -m "$MODEL" -clf $clf $cuda >> $results

printf "Test results: " >> $results
python simpletransformers/test_clf.py -d "$test_task1" "$test_task2" -m "$MODEL" -clf $clf $cuda >> $results

printf "\n" >> $results
