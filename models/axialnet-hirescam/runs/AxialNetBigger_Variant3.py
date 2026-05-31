import timeit

from src.run_experiment import DukeCTExperiment
from src.models import custom_models_bigger
from src.load_dataset import custom_datasets

if __name__=='__main__':
    experiments = {
        'AxialNetBigger_Variant3':{'custom_net':custom_models_bigger.AxialNetBigger_Variant3}}
    
    for key in experiments.keys():
        tot0 = timeit.default_timer()
        DukeCTExperiment(descriptor=key,
            base_results_dir = 'C:\\Users\\chamu\\D\\UOR\\FYP\\fyp\\models\\axialnet-hirescam\\results\\AxialNetBigger_Variant3',
            custom_net = experiments[key]['custom_net'],
            custom_net_args = {'n_outputs':80},
            loss_string = 'bce',
            loss_args = {},
            learning_rate = 1e-3,
            weight_decay = 1e-7,
            num_epochs=100, patience = 15,
            batch_size = 1, device = 0,
            data_parallel = False, model_parallel = False,
            use_test_set = False, task = 'train_eval',
            old_params_dir = '',
            dataset_class = custom_datasets.CTDataset_2019_10,
            dataset_args = {
                        'label_type_ld':'location_disease_0323',
                        'genericize_lung_labels':True,
                        'label_counts':{'mincount_heart':200,
                                    'mincount_lung':125},
                        'view':'axial',
                        'use_projections9':False,
                        'volume_prep_args':{
                                    'pixel_bounds':[-1000,800],
                                    'num_channels':3,
                                    'crop_type':'single',
                                    'selfsupervised':False,
                                    'from_seg':False},
                        'attn_gr_truth_prep_args':{
                                'dilate':None,
                                'downsamp_mode':None,
                                'small_square':10},  #small_square is 10 for this model variant! NOT 6!!!!
                        #Paths
                        'selected_note_acc_files':{'train':'C:\\Users\\chamu\\D\\UOR\\FYP\\fyp\\models\\axialnet-hirescam\\data\\RADChestCT_DEID\\predefined_subsets\\2020-01-10-imgtrain_random2000_DEID.csv',
                                                   'valid':'C:\\Users\\chamu\\D\\UOR\\FYP\\fyp\\models\\axialnet-hirescam\\data\\RADChestCT_DEID\\predefined_subsets\\2020-01-10-imgvalid_a_random1000_DEID.csv'},
                        'ct_scan_path':'C:\\Users\\chamu\\D\\UOR\\FYP\\fyp\\models\\axialnet-hirescam\\data\\RADChestCT_DEID',
                        'ct_scan_projections_path':'C:\\Users\\chamu\\D\\UOR\\FYP\\fyp\\models\\axialnet-hirescam\\data\\RADChestCT_DEID', #this path doesn't exist but I'm using non-projected data so it's ok.
                        'key_csvs_path':'C:\\Users\\chamu\\D\\UOR\\FYP\\fyp\\models\\axialnet-hirescam\\data\\RADChestCT_DEID/',
                        'segmap_path':'C:\\Users::chamu::D::UOR::FYP::fyp::models::axialnet-hirescam::data::RADChestCT_DEID/'})
        tot1 = timeit.default_timer()
        print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')