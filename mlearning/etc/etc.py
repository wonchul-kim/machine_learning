resume_info = {'data': 
                {'classes': ['NUBER_OK', 'LOT_OK', 'NUMBER_NG', 'LOT_NG'], 
                 'input_dir': '/DeepLearning/_athena_tests/datasets/rectangle1/split_dataset', 
                 'num_classes': 4, 
                 'output_dir': '/DeepLearning/etc/outputs'
                 }, 
                
                'dataset': 
                    {'annotation_format': 'labelme', 
                     'image_loading_mode': 'RGB', 
                     'dataset_type': None
                     }, 
                    
                'db': 
                    {'athena_db_server_url': '192.168.10.41:8001', 
                     'container_name': 'aiv-training-0', 
                     'exp_description': '', 
                     'project_description': '', 
                     'project_name': 'test', 
                     'server_host_name': 'wonchul', 
                     'server_ip': '192.168.11.177', 
                     'sub_project_name': 'det', 
                     'training_folder_name': 'aiv_training_0'
                     }, 
                'logging': 
                    {'LOGGING': True, 
                     'tf_log_level': 0, 
                     'log_stream_level': 'DEBUG', 
                     'log_file_level': 'DEBUG', 
                     'output_dir': '/DeepLearning/etc/outputs', 
                     'logs_dir': '/DeepLearning/etc/outputs/outputs/DETECTION/2024_03_16_14_19_59/train/logs', 
                     'configs_dir': '/DeepLearning/etc/outputs/outputs/DETECTION/2024_03_16_14_19_59/train/configs', 
                     'weights_dir': '/DeepLearning/etc/outputs/outputs/DETECTION/2024_03_16_14_19_59/train/weights', 
                     'val_dir': '/DeepLearning/etc/outputs/outputs/DETECTION/2024_03_16_14_19_59/train/val', 
                     'debug_dir': '/DeepLearning/etc/outputs/outputs/DETECTION/2024_03_16_14_19_59/train/debug', 
                     'dataset_dir': '/DeepLearning/etc/outputs/outputs/DETECTION/2024_03_16_14_19_59/train/dataset', 
                     'vis_dir': None, 
                     'patches_dir': None, 
                     'labels_dir': None, 
                     'monitor': True, 
                     'monitor_figs': True,
                     'monitor_csv': True,
                     'monitor_freq': 1,
                     'wb': False,
                     'aivdb': True,
                     'metisdb': False
                    },
                'model': 
                    {'anchor_t': 4,
                     'backbone': 'w6',
                     'box': 0.015,
                     'channel': 3,
                     'cls': 0.5,
                     'cls_pw': 1,
                     'fl_gamma': 0,
                     'height': 640,
                     'hyps': 'low',
                     'iou_t': 0.2,
                     'model_name': 'pytorch_yolov7',
                     'obj': 1,
                     'obj_pw': 1,
                     'seed_model': '',
                     'warmup_epochs': 3,
                     'width': 640,
                     'num_classes': 4
                    },
                'patch': 
                    {'bg_ratio_by_image': 0, 
                     'bg_start_train_epoch_by_image': 0,
                     'bg_start_val_epoch_by_image': 0,
                     'centric': False,
                     'height': 0,
                     'include_point_positive': True,
                     'num_involved_pixel': 10,
                     'overlap_ratio': 0.2,
                     'sliding': False,
                     'sliding_bg_ratio': 0,
                     'translate': 0,
                     'translate_range_height': 0,
                     'translate_range_width': 0,
                     'use_patch': False,
                     'width': 0
                    },
                'preprocess': 
                    {'normalize': {'type': 'max'}}, 
                'roi': 
                    {'from_json': False, 
                     'height': '', 
                     'top_left_x': '', 
                     'top_left_y': '', 
                     'use_roi': False, 
                     'width': ''
                    }, 
                'train': 
                    {'amp': False,
                     'anchors': '',
                     'batch_size': 8,
                     'debug_dataset_ratio': 0.2,
                     'device': 'gpu',
                     'device_ids': ['0', '1'],
                     'end_lr': 0.0001,
                     'epochs': 100,
                     'freeze': 0,
                     'image_weights': True,
                     'init_lr': 0.01,
                     'label_smoothing': 0.1,
                     'noautoanchor': False,
                     'num_hold': 0,
                     'num_warmup': 10,
                     'optimizer': 'sgd',
                     'patience': 0,
                     'scheduler': 'lambda',
                     'ml_framework': 'pytorch'
                    },
                'val': 
                    {'save_img_conf': 0.1,
                     'save_img_freq_epoch': 9, 
                     'save_img_iou': 0.25,
                     'save_img_ratio': 0.5,
                     'save_model': True,
                     'save_model_freq_epoch': 9
                    },
                'resume': 
                    {'use_resume': False,
                     'latest_output_dir': None,
                     'load_optimizer': False,
                     'load_learning_rate': False,
                     'create_directory': True,
                     'previous_output_dir': ''
                    }, 
                'augmentations': 
                    {'Albumentations': {}, 
                     'Customs': {'copy_paste': 0, 'degrees': 0, 'fliplr': 0.5, 
                                 'flipud': 0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'mixup': 0.05, 'mosaic': 0.5, 
                                 'paste_in': 0.05, 'perspective': 0, 'scale': 0.5, 'shear': 0, 'translate': 0.1}
                     }
        }

