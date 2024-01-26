### model_params.json
config file for model definition and optional weights load

**alphabet**: string of possible characters, the order of the classes is determined by the data markup

**fields**: list of class names, the order of the classes is determined by the data markup

**base_channels**: base number for channels for model layer blocks,
**dropout_rate**: dropout,
**field_threshold**: field detect threshhold,
**input_shape**: input image shape,
**weights_path \[optional\]**: path to pretrained weights for load


### train_config.json
config file for train definition

**TRAIN**
-----------------------------
**epochs**: epochs number,
**lr**: learning rate for SGD,
**momentum**: momentum for SGD,
**chars_weight_path**: path to char probabilities npy-file,
**fields_weight_path**: path to field probabilities npy-file,
**model_save_path**: save model weights path,
**save_condition**: save model mode:
    less_than_loss \[num: float\]- save weights if loss less then num;
    each_best_loss - save weights if loss less then better loss
    every_epoch - save weights every epoch

	
**TRAIN_DATA_GENERATOR**
-----------------------------
**img_path**: path to png-imgs,
**charmask_path**: path to char npy-masks,
**fieldmask_path**: path to field npy-masks,
**batch_size**: batch size,
**img_size**: image size
