# ICML 2021 State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

from utils import train_siamese, train_simclear, train_pca
from utils import infer_siamese, infer_simclear, infer_pca

#run shelf experiments no distractors
prefix="icml_box_"

models=["siamese","simclear","pca"]

train_datasets=["box_stacking_4b_vp_view_0_no_color","box_stacking_4b_vp_view_1_no_color","box_stacking_4b_vp_view_2_no_color","box_stacking_4b_vp_view_hard_mixed_no_color"]

#training
for model in models:
	for train_dataset in train_datasets:
		model_name=prefix + model + "_" + train_dataset
		if model=="siamese":
			train_siamese(model_name,train_dataset)
		if model=="simclear":
			train_simclear(model_name,train_dataset)
		if model=="pca":
			train_pca(model_name,train_dataset)


infer_datasets=["box_stacking_4b_vp_view_0_holdout_no_color","box_stacking_4b_vp_view_1_holdout_no_color","box_stacking_4b_vp_view_2_holdout_no_color","box_stacking_4b_vp_view_hard_mixed_holdout_no_color"]

#inference
for model in models:
	for train_dataset in train_datasets:
		model_name=prefix + model+ "_"  + train_dataset
		for infer_dataset in infer_datasets:		
			if model=="siamese":
				infer_siamese(model_name,infer_dataset)
			if model=="simclear":
				infer_simclear(model_name,infer_dataset)
			if model=="pca":
				infer_pca(model_name,infer_dataset)


