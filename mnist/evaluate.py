import argparse
import logging
import utils
import os
import numpy as np
from model import net
from model.data_loader import fetch_dataloader
import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

def evaluate(model, loss_fn, dataloader, metrics, params):
    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    for X, y in dataloader:
        if(params.cuda):
            #https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/8
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
        # convert to torch Variables
        X, y = Variable(X), Variable(y)
        # compute model output and loss
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        y_hat = y_hat.data.cpu().numpy()
        y = y.data.cpu().numpy()
        summary_batch = {metric: metrics[metric](y_hat, y) for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = fetch_dataloader(['test'], params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.LogisticRegression()
    if params.cuda:
        model = model.cuda()
    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)