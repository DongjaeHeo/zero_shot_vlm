# zero_shot_vlm

## Pre-processing
From the pre-processing stage to the evaluation stage, the dataset is divided using a 5-fold cross-validation setting based on the given seed.
### Per-Class Embedding Extraction 
```sh
python src/extracting_embeddings.py

```

### Train Binary Classifier per Class
```sh
# generate dataset
# generate all class
bash src/run_gen_crop_dataset.sh

# generate specific class
python src/generate_crop_dataset.py 0 Headstone_without_Mound 42 

# train binary classifier
# train all class
bash src/run_train_binary_classifier.sh

# train specific class
python src/train_binary_classifier.py 0 Headstone_without_Mound 42 

```

## Zero-shot Detection Evaluation
```sh
# Run all
bash src/run_eval_exp.sh

# Run Specific Evaluation
python src/main_eval.py -p topk_avg -t 0.91 -fcv
```