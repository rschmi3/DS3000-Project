#!/usr/bin/env bash

echo "Running all tumour classification configurations..."


for size in 128 256 340; do
    for seed in 3 42 75; do
        echo -e "\nImage size: $size, Random seed: $seed..."

        echo -e "\n------------------------------------------------\n"

        echo -e "Training conventional...\n"
        tumour-classifier --cuda --results-dir results --models-dir models --model conventional --image-size $size --random-state $seed --feature combined --lr 0.1 --train
        echo -e "Training conventional finished\n"

        echo -e "\n------------------------------------------------\n"

        echo -e "Training Neural...\n"
        tumour-classifier --cuda --results-dir results --models-dir models  --model Neural --image-size $size --random-state $seed --feature colour --epochs 30 --train
        echo -e "Training Neural finished\n"

        echo -e "\n------------------------------------------------\n"

        echo -e "Training NeuralSE...\n"
        tumour-classifier --cuda --results-dir results --models-dir models --model NeuralSE --image-size $size --random-state $seed --feature colour --epochs 40 --train
        echo -e "Training NeuralSE finished\n"

        echo -e "\n------------------------------------------------\n"

        echo -e "Training Neural_Aug...\n"
        tumour-classifier --cuda --results-dir results --models-dir models --model Neural_Aug --image-size $size --random-state $seed --feature colour --epochs 100 --train
        echo -e "Training Neural_Aug finished\n"

        echo -e "\n------------------------------------------------\n"

        echo -e "Training NeuralSE_Aug...\n"
        tumour-classifier --cuda --results-dir results --models-dir models --model NeuralSE_Aug --image-size $size --random-state $seed --feature colour --epochs 110 --train
        echo -e "Training NeuralSE_Aug finished\n"

        echo -e "\n------------------------------------------------\n"
    done
done
