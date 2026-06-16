sweep_train_hparams = {
        'num_epochs':   {'values': [ 1 ]},
        'batch_size':   {'values': [32,64]},
        'learning_rate': {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
        'disc_lr':      {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
        'weight_decay': {'values': [1e-4, 1e-5, 1e-6]},
        'step_size':    {'values': [5, 10, 30]},
        'optimizer':    {'values': ['adam']},
        'lr_decay':{'values': [0.5, 0.7, 0.9]},

}

sweep_alg_hparams = {
        'DANN': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'AdvSKM': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'CoDATS': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'CDAN': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'Deep_Coral': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'coral_wt':         {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
        },

        'DIRT': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'vat_loss_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'HoMM': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'hommd_wt':         {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},

        },

        'MMDA': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'coral_wt':         {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'DSAN': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},

        },

        'DDC': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},

        },
        
        'SASA': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'CoTMix': {
            'learning_rate':            {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'temporal_shift':           {'values': [5, 10, 15, 20, 30, 50]},
            'src_cls_weight':           {'distribution': 'uniform', 'min': 1e-1, 'max': 1},
            'mix_ratio':                {'distribution': 'uniform', 'min': 0.5, 'max': 0.99},
            'src_supCon_weight':        {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            'trg_cont_weight':          {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            'trg_entropy_weight':       {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
        },
        
        'uDAR': {
            # Paper: 0.001 - 0.0003
            'learning_rate':    {'distribution': 'log_uniform_values', 'min': 3e-4, 'max': 1e-3},
            
            # λ value (lambda): 0.18 - 0.45
            'cmmd_weight':      {'distribution': 'uniform', 'min': 0.18, 'max': 0.45},
            
            # α value (alpha): 0.55 - 0.75
            'temporal_alpha':   {'distribution': 'uniform', 'min': 0.55, 'max': 0.75},
            
            # σ value (sigma): 0.01 - 0.10
            'sigma':            {'distribution': 'uniform', 'min': 0.01, 'max': 0.10},
            
            # rθ value (r-theta): 5° - 45°
            'r_theta':          {'distribution': 'uniform', 'min': 5, 'max': 45},
            
            # --- Parameters from your code (not in table) ---
            # Using standard log-uniform ranges
            'kl_weight':        {'distribution': 'log_uniform_values', 'min': 1e-2, 'max': 10.0},
            'cmmd_gamma':       {'distribution': 'log_uniform_values', 'min': 1e-3, 'max': 1.0},
            'cmmd_lambda_reg':  {'distribution': 'log_uniform_values', 'min': 1e-3, 'max': 1.0}
        },
        
        'SWL_Adapt': {
            # Paper mentions 1e-3, but we can sweep around it
            'learning_rate':    {'values': [1e-3, 5e-4, 1e-4]},
            
            # Separate LR for the Weight Allocator
            'WA_lr':            {'values': [1e-3, 5e-4, 1e-4]},
            
            # Paper: "tuned under 80 and set to 3, 5, and 7"
            # We'll use a more modern range based on this.
            'WA_N_hid':         {'values': [5, 7, 8, 10,16]},
            
            # Paper: "confidence threshold ρ is tuned within [0.5...0.9]"
            'confidence_rate':  {'values': [0.5, 0.6, 0.7, 0.8, 0.9]},
            
            # Loss weight for target classification =
            'w_c_T':            {'distribution': 'log_uniform_values', 'min': 1e-2, 'max': 1.0}
        },

        # ----------------------------------------------------------------
        # DAAN — no original codebase; ranges derived from DANN baselines
        # in AdaTime HAR configs and the DAAN paper structure.
        # ----------------------------------------------------------------
        'DAAN': {
            'learning_rate':    {'values': [5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 0.5, 'max': 5.0},
            # global_loss_wt: paper uses ~0.05 default, sweep a wider range
            'global_loss_wt':   {'distribution': 'log_uniform_values', 'min': 0.01, 'max': 1.0},
            # local_loss_wt: paper uses ~0.01 default
            'local_loss_wt':    {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 0.5},
        },

        # ----------------------------------------------------------------
        # ACON — ranges from original ACON/main.py argparse defaults.
        # Paper defaults are all 1.0 except entropy_trade_off=0.01.
        # ----------------------------------------------------------------
        'ACON': {
            'learning_rate':        {'values': [5e-3, 1e-3, 5e-4]},
            'cls_trade_off':        {'distribution': 'uniform', 'min': 0.5, 'max': 5.0},
            'domain_trade_off':     {'distribution': 'uniform', 'min': 0.1, 'max': 5.0},
            'entropy_trade_off':    {'distribution': 'log_uniform_values', 'min': 1e-3, 'max': 0.1},
            'align_s_trade_off':    {'distribution': 'uniform', 'min': 0.1, 'max': 5.0},
            'align_t_trade_off':    {'distribution': 'uniform', 'min': 0.1, 'max': 5.0},
            'acon_disc_hid_dim':    {'values': [64, 128, 256]},
        },

        # ----------------------------------------------------------------
        # RAINCOAT — from Raincoat/configs/hparams.py HAR class.
        # Very few tunable params; paper uses fixed lr=5e-4 and 0.5/0.5 weights
        # for HAR. Sweep around those anchors.
        # ----------------------------------------------------------------
        'RAINCOAT': {
            'learning_rate':    {'values': [1e-3, 5e-4, 1e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 0.1, 'max': 2.0},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 0.1, 'max': 2.0},
        },

        # ----------------------------------------------------------------
        # SSSS_TSA — from SSSS_TSA/configs/hparams.py HAR_UCI class.
        # Original uses lr=1e-3, src_cls=1.613, domain=1.857, tau varies per
        # dataset (3 for 3-ch, 9-10 for 9-ch; paper rule: tau ≈ num_channels).
        # ----------------------------------------------------------------
        'SSSS_TSA': {
            'learning_rate':    {'values': [5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 0.5, 'max': 5.0},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 0.5, 'max': 5.0},
            'tau_temp':         {'values': [1, 3, 5, 9, 18]},
        },

        # ----------------------------------------------------------------
        # CLUDA — from CLUDA/main/train.py argparse defaults.
        # Original uses TCN with large batches (bs=2048, lr=5e-5); our port
        # uses CNN with bs=32, so lr is higher (~1e-3). Queue, momentum, and
        # temperature are architecture hparams from the CLUDA paper.
        # ----------------------------------------------------------------
        'CLUDA': {
            'learning_rate':    {'values': [5e-3, 1e-3, 5e-4, 1e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 0.5, 'max': 5.0},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 0.5, 'max': 5.0},
        },

}

