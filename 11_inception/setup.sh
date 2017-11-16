#!/bin/bash

get_data()
{
	data_dir=workspace
	rm -rf $data_dir || true
	mkdir -p $data_dir

	cd $data_dir
	curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
	tar xzf flower_photos.tgz
	cd -
}

get_script()
{
	rm retrain.py
	# need to match the version of source code to tensorflow's version
	# current version = r1.3
	wget https://raw.githubusercontent.com/tensorflow/tensorflow/r1.3/tensorflow/examples/image_retraining/retrain.py
}

main()
{
	get_data
	get_script
}

main $@
