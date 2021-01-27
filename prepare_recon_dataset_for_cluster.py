# ICML 2021 State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021
import pickle
from dataloader import preprocess_triplet_data_seed


def main():    

    dataset_names=["box_stacking_4b_vp_view_0_no_color",
                    "box_stacking_4b_vp_view_1_no_color",
                    "box_stacking_4b_vp_view_2_no_color",
                    "box_stacking_4b_vp_view_hard_mixed_no_color",
                    "2500_shelf_stacking",
                    "2500_shelf_stacking_ac_random_2_noac_swaps_0",
                    "2500_shelf_stacking_ac_random_1_noac_swaps_1",
                    "2500_shelf_stacking_ac_random_2_noac_swaps_1",
                    "2500_shelf_stacking_all_distractors",
                    "2500_shelf_stacking_all_distractors_ac_random_2_noac_swaps_0",
                    "2500_shelf_stacking_all_distractors_ac_random_1_noac_swaps_1",
                    "2500_shelf_stacking_all_distractors_ac_random_2_noac_swaps_1"
                    ]  

    seed=1122

    for dataset_name in dataset_names:
    	preprocess_triplet_data_seed(dataset_name+'.pkl',seed)
  

if __name__== "__main__":
  main()
