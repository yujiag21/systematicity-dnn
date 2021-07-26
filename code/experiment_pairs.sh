# Experiment parameters (modify these to run different experiments)

TASKS=('copy' 'different') # task1, task2

V_task1='abc' # vocabulary used only in task1 when training
V_task2='xyz' # vocabulary only used in task2 when training
V_both='' # vocabulary used in both tasks when training

MODEL='bert'
EPOCHS=5
EVAL_SPLIT=0.2
USE_CUDA=false # GPU/CPU

LM_BATCH_SIZE=16
CLF_BATCH_SIZE=16

min_len=3 # shortest sequences
max_len=3 # longest sequences

data_folder='data'
lm_folder='lm'
clf_folder='clf'

results='results/results.txt'
mkdir -p 'results'

echo "Tasks: ${TASKS[@]}" >> $results
echo "V_task1: $V_task1" >> $results
echo "V_task2: $V_task2" >> $results
echo "V_both: $V_both" >> $results
echo "Length: [$min_len, $max_len]" >> $results
echo "Model: $MODEL" >> $results

########################################################################################################################################################################

# Create datasets
python create_datasets.py -task "${TASKS[0]}" -voc "$V_task1$V_both" --min_len "$min_len" --max_len "$max_len" --save_folder "$data_folder" --eval_split $EVAL_SPLIT
python create_datasets.py -task "${TASKS[1]}" -voc "$V_task1" --min_len "$min_len" --max_len "$max_len" --save_folder "$data_folder" --eval_split 0
python create_datasets.py -task "${TASKS[1]}" -voc "$V_task2$V_both" --min_len "$min_len" --max_len "$max_len" --save_folder "$data_folder" --eval_split $EVAL_SPLIT
python create_datasets.py -task "${TASKS[0]}" -voc "$V_task2" --min_len "$min_len" --max_len "$max_len" --save_folder "$data_folder" --eval_split 0

# Train and test datasets
train_task1="$data_folder/${TASKS[0]}/$V_task1$V_both/train.txt"
eval_task1="$data_folder/${TASKS[0]}/$V_task1$V_both/eval.txt"
test_task1="$data_folder/${TASKS[0]}/$V_task2/all.txt"
train_task2="$data_folder/${TASKS[1]}/$V_task2$V_both/train.txt"
eval_task2="$data_folder/${TASKS[1]}/$V_task2$V_both/eval.txt"
test_task2="$data_folder/${TASKS[1]}/$V_task1/all.txt"

# count number of training and eval samples

printf "\nNumber of task1 training smaples: " >> $results
wc -l "$train_task1" >> $results
printf "\nNumber of task1 evaluation smaples: " >> $results
wc -l "$eval_task1" >> $results
printf "\nNumber of task2 training smaples: " >> $results
wc -l "$train_task2" >> $results
printf "\nNumber of task2 evaluation smaples: " >> $results
wc -l "$eval_task2" >> $results
printf "\n" >> $results

cuda=''
if $USE_CUDA
then
  cuda='--use_cuda'
fi

# Train language model on training data
printf "\nTraining language model\n"
python simpletransformers/train_lm.py --data "$train_task1" "$train_task2" -m "$MODEL" --batch_size $LM_BATCH_SIZE --epochs "$EPOCHS" --pairs $cuda

lm="$lm_folder/$MODEL"


# Train classifier on training data
printf "\nTraining classifier\n"
python simpletransformers/train_clf.py --train_data "$train_task1" "$train_task2" --eval_data "$eval_task1" "$eval_task2" -m "$MODEL" -lm $lm --epochs "$EPOCHS" --batch_size $CLF_BATCH_SIZE --pairs $cuda

clf="$clf_folder/$MODEL"


# Test trained classifier
eval_csv=pair_"$V_task1$V_task2$V_both"_eval_metric.csv
test_csv=pair_"$V_task1$V_task2$V_both"_test_metric.csv

printf "Evaluation results: " >> $results
python simpletransformers/test_clf.py -d "$eval_task1" "$eval_task2" -m "$MODEL" -clf $clf --pairs $cuda >> $results # sanity check: make sure eval results are same as in saved model

printf "Gather evaluation results on different epochs of the model..."
python simpletransformers/collect_clf_result.py -d "$eval_task1" "$eval_task2" -m "$MODEL" -clf $clf --pairs $cuda -rf "$eval_csv"

printf "Test results: " >> $results
python simpletransformers/test_clf.py -d "$test_task1" "$test_task2" -m "$MODEL" -clf $clf --pairs $cuda >> $results

printf "Gather test results on different epochs of the model..."
python simpletransformers/collect_clf_result.py -d "$test_task1" "$test_task2" -m "$MODEL" -clf $clf --pairs $cuda -rf "$test_csv"

printf "\n" >> $results
