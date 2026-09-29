import os
import logging
from pathlib import Path
import json
from datetime import datetime
from utils.util import ensure_dir

class ConfigParser:
    def __init__(self, config, modification=None, run_id=None):
        """
        Class de parse file JSON config va xu ly options tu command line.
        """
        self._config = config
        
        # ghi de cac thiet lap config bang cac modification tu CLI
        if modification:
            for key, val in modification.items():
                if isinstance(val, dict) and key in self._config:
                    self._config[key].update(val)
                else:
                    self._config[key] = val
                    
        # thiet lap thu muc luu checkpoints va logs
        save_dir = Path(self.config.get('save_dir', 'saved/models'))
        log_dir = Path(self.config.get('log_dir', 'saved/log'))
        
        name = self.config.get('name', 'FaceRecognition_Project')
        self._save_dir = save_dir / name
        self._log_dir = log_dir / name
        
        ensure_dir(self._save_dir)
        ensure_dir(self._log_dir)
        
        # luu lai file config.json vao thu muc luu tru de tracking
        with open(self._save_dir / 'config.json', 'w') as f:
            json.dump(self._config, f, indent=4)
            
        # cau hinh log levels
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, parser, args=None):
        """
        Khoi tao ConfigParser tu command line arguments.
        """
        if args is None:
            args = parser.parse_args()
            
        # doc file config
        msg_no_cfg = "Phai chi dinh file cau hinh config"
        assert args.config is not None, msg_no_cfg
        
        cfg_path = Path(args.config)
        with open(cfg_path, 'r') as f:
            config = json.load(f)
            
        # parse cac override tu CLI
        modification = {}
        if args.epochs:
            if 'trainer' not in modification:
                modification['trainer'] = {}
            modification['trainer']['epochs'] = args.epochs
        if args.batch_size:
            modification['batch_size'] = args.batch_size
        if args.lr:
            modification['learning_rate'] = args.lr
        if args.device:
            if 'trainer' not in modification:
                modification['trainer'] = {}
            modification['trainer']['accelerator'] = "gpu" if "gpu" in args.device or "cuda" in args.device else "cpu"
            
        return cls(config, modification, args.run_id)

    def get(self, key, default=None):
        return self._config.get(key, default)

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir
