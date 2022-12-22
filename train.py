from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
import matplotlib.pyplot as plt

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    #print(opt)

    logger = Logger("logs")
    #print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    #print(dataset)

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    loss_graph=[]
    metrica_precision_valid = []
    metrica_recall_valid = []
    metrica_mAP_valid = []
    metrica_f1_valid = []
    epocas_metricas_valid = []

    metrica_precision_train = []
    epocas_metricas_train = []
    epocas=[]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            precision_med = 0
            contador = 0

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                #print("test")
                #print(row_metrics)

                if (metric=="precision"):
                    precision_med = precision_med + ((float(row_metrics[0])+float(row_metrics[1])+float(row_metrics[2]))/3)

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            
            metrica_precision_train.append(precision_med)
            epocas_metricas_train.append(epoch)
            loss_graph.append(loss.item())
            epocas.append(epoch)

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)
        #loss_graph.append(loss.item())
        #epocas.append(epoch)
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]

            #print(evaluation_metrics[0][1])
            #print(evaluation_metrics[1][1])
            #print(evaluation_metrics[2][1])
            #print(evaluation_metrics[3][1])
            #print(epoch)
            metrica_precision_valid.append(evaluation_metrics[0][1])
            metrica_recall_valid.append(evaluation_metrics[1][1])
            metrica_mAP_valid.append(evaluation_metrics[2][1])
            metrica_f1_valid.append(evaluation_metrics[3][1])
            epocas_metricas_valid.append(epoch)

            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch == (opt.epochs-1):
            torch.save(model.state_dict(), f"model_2_number_pti_%d.pth" % epoch)
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
    

    #plt.plot(epocas,loss_graph, label="Loss")
    #plt.grid(True)
    #plt.legend()
    #plt.show()

    axis_color = 'lightgoldenrodyellow'
    fig1 = plt.figure("Metricas")

    ax1 = fig1.add_subplot(224)
    plt.title("Precision Valid")
    plt.xlabel("epocas", fontsize = 7)
    plt.ylabel("Precision", fontsize = 7)
    plt.grid(True)

    ax2 = fig1.add_subplot(223)
    plt.title("Loss Valid")
    plt.xlabel("epocas", fontsize = 7)
    plt.ylabel("Loss", fontsize = 7)
    plt.grid(True)

    ax3 = fig1.add_subplot(222)
    plt.title("Recall Valid")
    plt.xlabel("epocas", fontsize = 7)
    plt.ylabel("Recall", fontsize = 7)
    plt.grid(True)

    ax4 = fig1.add_subplot(221)
    plt.title("f1 Valid")
    plt.xlabel("epocas", fontsize = 7)
    plt.ylabel("f1", fontsize = 7)
    plt.grid(True)

    fig1.subplots_adjust(left=0.05, bottom=0.25, hspace=1) 

    [line1] = ax1.plot(epocas_metricas_valid, metrica_precision_valid, linewidth=1, color='r')
    [line2] = ax1.plot(epocas_metricas_train, metrica_precision_train, linewidth=1, color='orange')
    [line3] = ax2.plot(epocas, loss_graph, linewidth=1)
    [line4] = ax3.plot(epocas_metricas_valid, metrica_recall_valid, linewidth=1)
    [line5] = ax4.plot(epocas_metricas_valid, metrica_f1_valid, linewidth=1)

    plt.show()
