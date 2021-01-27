#!/bin/bash 

# ICML 2021 State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

#Box stacking
#python train_recon_models.py --exp_vae='icml_ae_box_stacking_4b_vp_view_0_no_color_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_box_stacking_4b_vp_view_1_no_color_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_box_stacking_4b_vp_view_2_no_color_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_box_stacking_4b_vp_view_hard_mixed_no_color_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='icml_ae_box_stacking_4b_vp_view_0_no_color_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_box_stacking_4b_vp_view_1_no_color_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_box_stacking_4b_vp_view_2_no_color_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_box_stacking_4b_vp_view_hard_mixed_no_color_w1000' --cuda=True 
#
python train_recon_models.py --exp_vae='icml_iros_vae_box_stacking_4b_vp_view_0_no_color_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_box_stacking_4b_vp_view_1_no_color_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_box_stacking_4b_vp_view_2_no_color_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_box_stacking_4b_vp_view_hard_mixed_no_color_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='icml_iros_vae_box_stacking_4b_vp_view_0_no_color' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_box_stacking_4b_vp_view_1_no_color' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_box_stacking_4b_vp_view_2_no_color' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_box_stacking_4b_vp_view_hard_mixed_no_color' --cuda=True 
#
#
##shelf arrangment
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_ac_random_2_noac_swaps_0_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_ac_random_1_noac_swaps_1_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_ac_random_2_noac_swaps_1_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_all_distractors_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_all_distractors_ac_random_2_noac_swaps_0_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_all_distractors_ac_random_1_noac_swaps_1_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_all_distractors_ac_random_2_noac_swaps_1_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_ac_random_2_noac_swaps_0_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_ac_random_1_noac_swaps_1_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_ac_random_2_noac_swaps_1_w1000' --cuda=True 
#
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_all_distractors_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_all_distractors_ac_random_2_noac_swaps_0_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_all_distractors_ac_random_1_noac_swaps_1_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='icml_ae_shelf_stacking_all_distractors_ac_random_2_noac_swaps_1_w1000' --cuda=True 
#
#
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_ac_random_2_noac_swaps_0_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_ac_random_1_noac_swaps_1_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_ac_random_2_noac_swaps_1_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_all_distractors_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_all_distractors_ac_random_2_noac_swaps_0_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_all_distractors_ac_random_1_noac_swaps_1_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_all_distractors_ac_random_2_noac_swaps_1_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_ac_random_2_noac_swaps_0' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_ac_random_1_noac_swaps_1' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_ac_random_2_noac_swaps_1' --cuda=True 
#
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_all_distractors_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_all_distractors_ac_random_2_noac_swaps_0' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_all_distractors_ac_random_1_noac_swaps_1' --cuda=True 
#python train_recon_models.py --exp_vae='icml_iros_vae_shelf_stacking_all_distractors_ac_random_2_noac_swaps_1' --cuda=True 
