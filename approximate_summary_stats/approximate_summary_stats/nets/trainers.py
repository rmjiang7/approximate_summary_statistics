import torch
from tqdm.auto import tqdm
import numpy as np

class DataLoaderTrainer(object):

    @staticmethod
    def generate_train_valid(ts, params, batch_size, train_percent, seed = None, device = 'cpu'):

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(ts, requires_grad = False).float(),
            torch.tensor(params, requires_grad = False).float()
        )

        n_train = int(ts.shape[0] * train_percent)
        n_valid = ts.shape[0] - n_train
        if seed is not None:
            train_set, val_set = torch.utils.data.random_split(
                                    dataset,
                                    [n_train, n_valid],
                                    torch.Generator().manual_seed(seed))
        else:
            train_set, val_set = torch.utils.data.random_split(
                                    dataset,
                                    [n_train, n_valid])


        train_loader = torch.utils.data.DataLoader(
                            train_set,
                            batch_size = batch_size,
                            shuffle = False
        )

        val_loader = torch.utils.data.DataLoader(
                            val_set,
                            batch_size = n_valid,
                            shuffle = False
        )

        return train_loader, val_loader

    @staticmethod
    def train(model,
              train_loader,
              val_loader,
              loss_fn,
              optimizer,
              scheduler = None,
              n_epochs = 3000,
              patience = np.inf,
              device = 'cpu'):
        
        if device == 'gpu':
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        model.to(device)

        epochs_no_improve = 0
        best_epoch = np.inf
        epoch_val_loss = []

        outer = tqdm(range(n_epochs), desc = 'Epoch', position = 0)
        for i in outer:
            for train_batch in tqdm(train_loader, total=len(train_loader), desc = 'Batch', position = 1, leave = False):

                optimizer.zero_grad()
                x,y = train_batch[0].to(device), train_batch[1].to(device)
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_val_loss = 0.0
            for vd in val_loader:
                x,y = vd[0].to(device), vd[1].to(device)
                output = model(x)
                val_loss = loss_fn(output,y)
                total_val_loss += val_loss.item()
            epoch_val_loss.append(total_val_loss)

            if total_val_loss >= best_epoch:
                epoch_no_improve += 1
                if epoch_no_improve == patience:
                    print("Early Stopping")
                    break
            else:
                best_epoch = total_val_loss
                epoch_no_improve = 0
            outer.set_description("Epoch Loss : %f" % total_val_loss)

        return epoch_val_loss
