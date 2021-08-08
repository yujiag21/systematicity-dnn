# Experiment parameters (modify these to run different experiments)

V_task1='abc' # vocabulary used only in task1 when training
V_task2='x' # vocabulary only used in task2 when training
V_both='' # vocabulary used in both tasks when training


DIFFERENT_TASK=('different' '0')
COPY_TASK=('copy' '1')
REVERSE_TASK=('reverse' '2')
DUMMY_TASK=('dummy' '3')


task_1=(${COPY_TASK[0]} ${COPY_TASK[1]})
task_2=(${DUMMY_TASK[0]} ${DUMMY_TASK[1]})



MODEL='bert'
EPOCHS=5
EVAL_SPLIT=0.2
USE_CUDA=false # GPU/CPU

LM_BATCH_SIZE=16
CLF_BATCH_SIZE=16

data_folder='data'
lm_folder='lm'
enc_dec_folder='enc_dec'


results='results/results_enc_dec_dummy.txt'
mkdir -p 'results'

# sequence lengths
min_len=1 # shortest sequence
max_len=3 # longest sequences

echo "task 1: ${task_1[@]}" >> $results
echo "task 2: ${task_2[@]}" >> $results

echo "V_task1: $V_task1" >> $results
echo "V_task2: $V_task2" >> $results
echo "V_both: $V_both" >> $results
echo "Length: [$min_len, $max_len]" >> $results

echo "Model: $MODEL" >> $results

echo 'Create dataset...'
python create_datasets.py -task ${task_1[0]} -voc "$V_task1$V_both" --min_len "$min_len" \
--max_len "$max_len" --repeat "$V_task1$V_both" --save_folder "$data_folder" --save_suffix\
 ${task_1[1]}${task_1[0]} --eval_split $EVAL_SPLIT --src_prefix ${task_1[1]}

train_task1="$data_folder/${task_1[0]}/${V_task1}${V_both}_${task_1[1]}${task_1[0]}/train.txt"
eval_task1="$data_folder/${task_1[0]}/${V_task1}${V_both}_${task_1[1]}${task_1[0]}/eval.txt"
test_task1="$data_folder/${task_1[0]}/${V_task1}_${task_1[1]}${task_1[0]}/all.txt"

wc ${test_task1} > xxx
read lines words characters filename < xxx

python create_datasets.py -task "${task_2[0]}" -voc "x" --min_len "1" \
--max_len "1" --save_folder "$data_folder" --save_suffix\
 ${task_2[1]}${task_2[0]} --eval_split $EVAL_SPLIT --src_prefix ${task_2[1]} --dummy_min_size $lines

train_task2="$data_folder/${task_2[0]}/${V_task2}${V_both}_${task_2[1]}${task_2[0]}/train.txt"
eval_task2="$data_folder/${task_2[0]}/${V_task2}${V_both}_${task_2[1]}${task_2[0]}/eval.txt"
test_task2="$data_folder/${task_2[0]}/${V_task2}_${task_2[1]}${task_2[0]}/all.txt"


printf "\nNumber of task1 training smaples: " >> $results
wc -l "$train_task1" >> $results
printf "\nNumber of task1 evaluation smaples: " >> $results
wc -l "$eval_task1" >> $results
printf "\nNumber of task2 training smaples: " >> $results
wc -l "$train_task2" >> $results
printf "\nNumber of task2 evaluation smaples: " >> $results
wc -l "$eval_task2" >> $results
printf "\n" >> $results



# Train language model on training data
printf "\nTraining language model\n"
python simpletransformers/train_lm.py --data "$train_task1" "$train_task2" -m "$MODEL" --batch_size $LM_BATCH_SIZE --epochs "$EPOCHS" $cuda

lm="$lm_folder/$MODEL"

echo 'Train encoder decoder...'
python simpletransformers/train_enc_dec.py  --train_data "$train_task1" "$train_task2" \
-m "$MODEL" --batch_size $LM_BATCH_SIZE --epochs "$EPOCHS" $cuda -lm $lm

enc_dec="$enc_dec_folder/$MODEL"

printf "Test results: " >> $results
python simpletransformers/test_enc_dec.py -d "$test_task1" "$test_task2" -m "$MODEL" -enc_dec $enc_dec $cuda >> $results

printf "\n" >> $results