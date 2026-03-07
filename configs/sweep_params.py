sweep_train_hparams = {
        'num_epochs':   {'values': [ 5 ]},
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
        
}

