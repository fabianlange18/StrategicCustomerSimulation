import config

def update_dirs():
    config.plot_dir = f"./results/{config.run_name}/plots/"
    config.tb_dir = f"./tensorboard/{config.run_name}/"
    config.summary_dir = f"./results/{config.run_name}/"