import argparse
import utils
import numpy as np
import os
import logging
import torch
from torch.autograd import Variable
from tqdm import tqdm
import model.data_loader
from model.data_loader import fetch_dataloader
from model import net
from torch import optim
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data',help="Directory containing the dataset")
parser.add_argument('--model-dir',  default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")

def train(model, optimizer, loss_fcn, dataloader, metrics, params):
    # set model to training mode
    model.train()
    summ = []
    loss_avg = utils.RunningAverage()
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (X, y) in enumerate(dataloader):
            if(params.cuda):
                #https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/8
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # convert to torch Variables
            X, y = Variable(X), Variable(y)
            # compute model output and loss
            y_hat = model(X)
            loss = loss_fcn(y_hat, y)
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                y_hat = y_hat.data.cpu().numpy()
                y = y.data.cpu().numpy()
                summary_batch = {metric: metrics[metric](y_hat, y) for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)
            
            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    if restore_file:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        #compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)
        val_metrics = evaluate(model,loss_fn, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = fetch_dataloader(
        ['train', 'val'], params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.LogisticRegression(params.input_dim, params.output_dim).cuda() if params.cuda else net.LogisticRegression(params.input_dim, params.output_dim)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)