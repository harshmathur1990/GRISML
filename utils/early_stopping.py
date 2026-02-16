import torch


class EarlyStopping:
    def __init__(self, patience, min_delta=0.0, path="checkpoint.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path

        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss, model):
        """
        Returns True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

            # --- save underlying model (handles DataParallel safely)
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, self.path)

            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
