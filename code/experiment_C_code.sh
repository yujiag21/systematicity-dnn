
MODEL='bert'
EPOCHS=5
EVAL_SPLIT=0.2
USE_CUDA=false # GPU/CPU

LM_BATCH_SIZE=16
CLF_BATCH_SIZE=16

input_data_file='data_processor/processed_data.csv'
data_folder='data'
lm_folder='lm'
clf_folder='clf'
dataset_name='C_variable_vul'

results='results/results_C.txt'
mkdir -p 'results'

task='CCode'

echo "Model: $MODEL" >> $results

########################################################################################################################################################################

# Create datasets
python data_processor/split_datasets.py --dataset_name "$dataset_name" --input_processed_data_file "$input_data_file" -task "$task"  --save_folder "$data_folder" --eval_split $EVAL_SPLIT

# Train and test datasets
train_task1="$data_folder/${task}/${dataset_name}/train.txt"
eval_task1="$data_folder/${task}/${dataset_name}/eval.txt"
test_task1="$data_folder/${task}/${dataset_name}/all.txt"

cuda=''
if $USE_CUDA
then
  cuda='--use_cuda'
fi


lm="$lm_folder/$MODEL"

# Train classifier on training data
printf "\nTraining classifier\n"
python simpletransformers/train_clf.py --train_data "$train_task1" --eval_data "$eval_task1" -m "$MODEL" -lm $lm --epochs "$EPOCHS" --batch_size $CLF_BATCH_SIZE $cuda
clf="$clf_folder/$MODEL"


printf "Test results: " >> $results
python simpletransformers/test_clf.py -d "$test_task1" -m "$MODEL" -clf $clf $cuda >> $results

printf "\n" >> $results
