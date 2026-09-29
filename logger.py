import logging
import logging.config
from pathlib import Path
from utils.util import read_json

def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Thiet lap cau hinh logging
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # thay doi log path de luu vao dung save_dir
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                # dam bao thu muc log ton tai
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                handler['filename'] = str(Path(save_dir) / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print(f"Warning: khong tim thay file cau hinh logging tai '{log_config}'. Dung basicConfig.")
        logging.basicConfig(level=default_level)
