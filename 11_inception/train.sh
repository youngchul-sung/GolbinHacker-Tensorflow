#!/bin/bash

python retrain.py \
	--bottleneck_dir=./workspace/bottlenecks \
	--model_dir=./workspace/inception \
	--output_graph=./workspace/flowers_graph.pb \
	--output_labels=./workspace/flowers_labels.txt \
	--image_dir ./workspace/flower_photos \
	--how_many_training_steps 1000
