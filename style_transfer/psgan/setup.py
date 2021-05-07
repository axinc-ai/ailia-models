from psgan.config import get_config


def setup_config(args):
    config = get_config()
    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)
    config.freeze()
    return config
