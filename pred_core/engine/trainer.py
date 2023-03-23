import datetime
import logging
import time
import torch
import numpy as np
import os

from pred_core.utils.metric_logger import MetricLogger


def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        device,
        checkPointer,
        arguments
):
    model = model.train()
    meters = MetricLogger(delimiter="  ")
    scaler = data_loader['scaler']

    logger = logging.getLogger("pred_core.trainer")
    logger.info("Start training")

    num_batches = data_loader['train_loader'].num_batch
    logger.info("num_batches:{}".format(num_batches))

    total_epoch = arguments['epochs']
    epoch_nums = arguments['num_epoch']
    batches_seen = num_batches * arguments['num_epoch']
    end = time.time()

    for epoch_num in range(epoch_nums, total_epoch):
        data_time = time.time() - end

        losses = []

        train_iterator = data_loader['train_loader'].get_iterator()

        for dict_result, (x, y) in enumerate(train_iterator):
            optimizer.zero_grad()
            x, y = prepare_data(x, y)
            x = x.to(device)
            y = y.to(device)

            output = model(x, y, batches_seen)

            loss = compute_loss(scaler, y, output)

            losses.append(loss.item())

            batches_seen += 1

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

        epoch_time = time.time() - end
        end = time.time()
        meters.update(time=epoch_time, data=data_time)
        mean_loss = np.mean(losses)
        meters.update(loss=mean_loss)
        eta_seconds = meters.time.global_avg * (total_epoch - epoch_num)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        scheduler.step()
        logger.info(
            meters.delimiter.join(
                [
                    "eta: {eta}",
                    "iter: {iter}",
                    "{meters}",
                    "lr: {lr:.6f}",
                    "max mem: {memory:.0f}",
                ]
            ).format(
                eta=eta_string,
                iter=epoch_num,
                meters=str(meters),
                lr=optimizer.param_groups[0]["lr"],
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            )
        )

        if epoch_num % arguments['test_period'] == 0:
            test_loss, dict_result = evaluate(model, data_loader, scaler, device)
            store_prediction_data(dict_result, "/mnt/develop/PycharmProjects/blue_green_alage_prediction/output")
            logger.info('Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                        .format(epoch_num, total_epoch, batches_seen,
                                np.mean(losses), test_loss, optimizer.param_groups[0]["lr"]))

    checkPointer.save("model_final", **arguments)


def store_prediction_data(dict_result, path):
    pred = dict_result['prediction'][0]
    truth = dict_result['truth'][0]
    np.save(os.path.join(path, 'pred.npy'), pred)
    np.save(os.path.join(path, 'truth.npy'), truth)


def prepare_data(x, y):
    x, y = get_x_y(x, y)
    x, y = get_x_y_in_correct_dims(x, y)
    return x, y


def get_x_y(x, y):
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x = x.permute(1, 0, 2, 3)
    y = y.permute(1, 0, 2)
    return x, y


def get_x_y_in_correct_dims(x, y):
    batch_size = x.size(1)
    seq_len = x.shape[0]
    x = x.view(seq_len, batch_size, -1)
    return x, y


def compute_loss(scaler, y_true, y_predicted):
    y_true = scaler.inverse_transform(y_true, 'y')
    y_predicted = scaler.inverse_transform(y_predicted, 'y')
    return masked_mae_loss(y_predicted, y_true)


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def save_model(epoch, model, logger):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    config = dict()
    config['model_state_dict'] = model.state_dict()
    config['epoch'] = epoch
    torch.save(config, 'models/epo%d.tar' % epoch)
    logger.info("Saved model at {}".format(epoch))
    return 'models/epo%d.tar' % epoch


def evaluate(model,
             data_loader,
             scaler,
             device
             ):
    with torch.no_grad():
        model = model.eval()
        model = model.to(device)
        test_iterator = data_loader['train_loader'].get_iterator()
        losses = []

        y_truths = []
        y_preds = []
        for _, (x, y) in enumerate(test_iterator):
            x, y = prepare_data(x, y)
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = compute_loss(scaler, y, output)
            losses.append(loss.item())

            y_truths.append(y.cpu())
            y_preds.append(output.cpu())
        mean_loss = np.mean(losses)
        y_preds = np.concatenate(y_preds, axis=1)
        y_truths = np.concatenate(y_truths, axis=1)
        y_truths_scaled = []
        y_preds_scaled = []
        for t in range(y_preds.shape[0]):
            y_truth = scaler.inverse_transform(y_truths[t], 'y')
            y_pred = scaler.inverse_transform(y_preds[t], 'y')
            y_truths_scaled.append(y_truth)
            y_preds_scaled.append(y_pred)

        return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}
