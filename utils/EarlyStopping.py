


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        #print(f"Best: {self.best_loss} Current: {val_loss} Counter: {self.counter}")
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            #print(f"Metric improving")
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            #print(f"Metric not improving")
            self.counter += 1
            # print(f"\nINFO: Early stopping counter {self.counter} of {self.patience}\n")
            if self.counter >= self.patience:
                print('\nINFO: Early stopping\n')
                self.early_stop = True