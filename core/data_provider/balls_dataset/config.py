
def set_dataset_args(cfg, args, dataset_name, model_name, mode='train'):
    ds_kwargs = dict()
    loader_kwargs = dict()
    if model_name == 'Obsurf':
        ds_kwargs['load_ab'] = cfg['data']['load_ab']
        ds_kwargs['jitter'] = cfg['data']['jitter']
        ds_kwargs['path'] = args.data_dir
        loader_kwargs['batch_size'] = cfg['training']['batch_size']
        loader_kwargs['num_workers'] = cfg['training']['num_workers']
        if mode == 'train':
            loader_kwargs['shuffle'] = True
        else:
            loader_kwargs['shuffle'] = False
        loader_kwargs['pin_memory'] = False
        loader_kwargs['persistent_workers'] = True
    elif model_name == 'SAVi':
        ds_kwargs['load_ab'] = cfg['data']['load_ab']
        ds_kwargs['load_cd'] = False
        ds_kwargs['jitter'] = cfg['data']['jitter']
        ds_kwargs['path'] = args.data_dir
        ds_kwargs['sampling_mode'] = 'fix'
        loader_kwargs['batch_size'] = cfg['training']['batch_size']
        loader_kwargs['num_workers'] = cfg['training']['num_workers']
        if mode == 'train':
            loader_kwargs['shuffle'] = True
        else:
            loader_kwargs['shuffle'] = False
        loader_kwargs['pin_memory'] = False
        loader_kwargs['persistent_workers'] = True
    elif model_name == 'VisionDynamics':
        ds_kwargs['load_ab'] = cfg['data']['load_ab']
        ds_kwargs['load_cd'] = False
        ds_kwargs['jitter'] = cfg['data']['jitter']
        ds_kwargs['path'] = args.data_dir
        ds_kwargs['image_path'] = args.img_dir
        ds_kwargs['sampling_mode'] = args.sampling_mode  # rand, fix, full
        loader_kwargs['batch_size'] = cfg['training']['batch_size']
        loader_kwargs['num_workers'] = cfg['training']['num_workers']
        if mode == 'train':
            loader_kwargs['shuffle'] = True
        else:
            loader_kwargs['shuffle'] = False
        loader_kwargs['pin_memory'] = False
        loader_kwargs['persistent_workers'] = True
    return ds_kwargs, loader_kwargs
