class SimpleWriter:
    def __init__(self, logger=None, config=None, run_name="default", log_checkpoints=False):
        self.logger = logger
        self.config = config
        self.run_name = run_name
        self.log_checkpoints = bool(log_checkpoints)
        self._step = 0
        self._mode = "train"

    def set_step(self, step, mode="train"):
        self._step = int(step)
        self._mode = mode

    def add_scalar(self, name, value):
        if self.logger is not None:
            self.logger.debug(f"[{self._mode} step={self._step}] {name}: {value}")

    def add_checkpoint(self, filename, root_dir):
        del root_dir
        if self.logger is not None:
            self.logger.info(f"Checkpoint logged: {filename}")
