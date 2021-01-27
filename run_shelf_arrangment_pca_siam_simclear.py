# ICML 2021 State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

from utils import train_siamese, train_simclear, train_pca
from utils import infer_siamese, infer_simclear, infer_pca



#run shelf experiments no distractors
prefix="icml_shelf_"

models=["siamese","simclear","pca"]

train_datasets=["2500_shelf_stacking","2500_shelf_stacking_ac_random_2_noac_swaps_0","2500_shelf_stacking_ac_random_1_noac_swaps_1","2500_shelf_stacking_ac_random_2_noac_swaps_1"]

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


infer_datasets=["2500_shelf_stacking","2500_shelf_stacking_holdout"]

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



#now we train with distractos
train_datasets=["2500_shelf_stacking_all_distractors","2500_shelf_stacking_all_distractors_ac_random_2_noac_swaps_0","2500_shelf_stacking_all_distractors_ac_random_1_noac_swaps_1","2500_shelf_stacking_all_distractors_ac_random_2_noac_swaps_1"]

#training
for model in models:
	for train_dataset in train_datasets:
		model_name=prefix + model+ "_" + train_dataset
		if model=="siamese":
			train_siamese(model_name,train_dataset)
		if model=="simclear":
			train_simclear(model_name,train_dataset)
		if model=="pca":
			train_pca(model_name,train_dataset)


infer_datasets=["2500_shelf_stacking_all_distractors","2500_shelf_stacking_all_distractors_holdout"]

#inference
for model in models:
	for train_dataset in train_datasets:
		model_name=prefix + model+ "_" + train_dataset
		for infer_dataset in infer_datasets:		
			if model=="siamese":
				infer_siamese(model_name,infer_dataset)
			if model=="simclear":
				infer_simclear(model_name,infer_dataset)
			if model=="pca":
				infer_pca(model_name,infer_dataset)
