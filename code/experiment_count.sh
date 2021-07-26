# Experiment parameters (modify these to run different experiments)

V_task1='abc' # vocabulary used only in task1 when training
V_task2='xyz' # vocabulary only used in task2 when training
V_both='' # vocabulary used in both tasks when training

# sequence lengths per task: make these *shorter for task1*
min_len_task1=1 # shortest sequences for task1
max_len_task1=2 # longest sequences for task1
min_len_task2=3 # shortest sequences for task2
max_len_task2=4 # longest sequences for task2

MODEL='bert'
EPOCHS=5
EVAL_SPLIT=0.2
USE_CUDA=false # GPU/CPU

LM_BATCH_SIZE=16
CLF_BATCH_SIZE=16

data_folder='data'
lm_folder='lm'
clf_folder='clf'

results='results/results_count.txt'
mkdir -p 'results'

echo "$V_task1 $V_both [$min_len_task1, $max_len_task1]" >> $results
echo "$V_task2 $V_both [$min_len_task2, $max_len_task2]" >> $results
echo "Model: $MODEL" >> $results

########################################################################################################################################################################

# Create datasets
python create_datasets.py -task count -voc "$V_task1$V_both" --min_len "$min_len_task1" --max_len "$max_len_task1" --save_folder "$data_folder" --eval_split $EVAL_SPLIT
max_train_size=$(< "$data_folder/count/${V_task1}${V_both}_$min_len_task1-$max_len_task1/all.txt" wc -l) # get size of task1 dataset, fix to size of the task2 to ensure balance
python create_datasets.py -task count -voc "$V_task1" --min_len "$min_len_task2" --max_len "$max_len_task2" --save_folder "$data_folder" --eval_split 0
python create_datasets.py -task count -voc "$V_task2$V_both" --min_len "$min_len_task2" --max_len "$max_len_task2" --save_folder "$data_folder" --eval_split $EVAL_SPLIT --max_size "$max_train_size"
python create_datasets.py -task count -voc "$V_task2" --min_len "$min_len_task1" --max_len "$max_len_task1" --save_folder "$data_folder" --eval_split 0

# Train and test datasets
train_task1="$data_folder/count/${V_task1}${V_both}_$min_len_task1-$max_len_task1/train.txt"
eval_task1="$data_folder/count/${V_task1}${V_both}_$min_len_task1-$max_len_task1/eval.txt"
test_task1="$data_folder/count/${V_task2}_$min_len_task1-$max_len_task1/all.txt"
train_task2="$data_folder/count/${V_task2}${V_both}_$min_len_task2-$max_len_task2/train.txt"
eval_task2="$data_folder/count/${V_task2}${V_both}_$min_len_task2-$max_len_task2/eval.txt"
test_task2="$data_folder/count/${V_task1}_$min_len_task2-$max_len_task2/all.txt"

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
eval_csv=count_"$V_task1$V_task2$V_both"_eval_metric.csv
test_csv=count_"$V_task1$V_task2$V_both"_test_metric.csv

printf "Evaluation results: " >> $results
python simpletransformers/test_clf.py -d "$eval_task1" "$eval_task2" -m "$MODEL" -clf $clf $cuda >> $results

printf "Gather evaluation results on different epochs of the model..."
python simpletransformers/collect_clf_result.py -d "$eval_task1" "$eval_task2" -m "$MODEL" -clf $clf $cuda -rf "$eval_csv"

printf "Test results: " >> $results
python simpletransformers/test_clf.py -d "$test_task1" "$test_task2" -m "$MODEL" -clf $clf $cuda >> $results

printf "Gather test results on different epochs of the model..."
python simpletransformers/collect_clf_result.py -d "$test_task1" "$test_task2" -m "$MODEL" -clf $clf $cuda -rf "$test_csv"

printf "\n" >> $results