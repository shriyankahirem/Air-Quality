import torch
import argparse

from tqdm import tqdm

from geo_attn.modules import Geo_Attention_Model
from utils.data import load_data


parser = argparse.ArgumentParser(description='Train Geo-Attention Model')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

def config(args):
    args.save_path = './weights/geo_attn.pth'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args


def train(args):
    train_loader, val_loader, test_loader = load_data(batch_size=args.batch_size)

    model = Geo_Attention_Model().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.MSELoss()

    print("Model Trained on {}".format(args.device))
    best_val_loss = 1e10
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            locs, readings, target_loc, target_reading = batch
            locs, readings, target_loc, target_reading = locs.to(args.device), readings.to(args.device), target_loc.to(args.device), target_reading.to(args.device)
            optimizer.zero_grad()
            pred = model((locs, readings, target_loc))
            loss = criterion(pred, target_reading)
            loss.backward()
            train_loss += loss.cpu().detach().item()
            optimizer.step()
        train_loss /= len(train_loader)
        scheduler.step()

        # evaluate the model
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for batch in val_loader:
                locs, readings, target_loc, target_reading = batch
                locs, readings, target_loc, target_reading = locs.to(args.device), readings.to(args.device), target_loc.to(args.device), target_reading.to(args.device)
                pred = model((locs, readings, target_loc))
                loss = criterion(pred, target_reading)
                val_loss += loss.cpu().detach().item()
            val_loss /= len(val_loader)
        print('Epoch: {}/{}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch, args.epochs, train_loss, val_loss))

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)

def main(args):
    args = config(args)
    train(args)


if __name__ == '__main__':
    main(args)