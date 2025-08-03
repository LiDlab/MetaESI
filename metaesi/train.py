import os
import copy
import math
import torch
from torch import nn, optim
import numpy as np
from .model import MetaESI
from .utils import try_gpu
import learn2learn as l2l



def meta_train_MetaESI(Dataset, Feature, episodes, gpu=0):

    # Parameters Initialization and Model Setup
    meta_lr = 1e-4
    fast_lr = 0.05

    model = MetaESI(1280).to(device=try_gpu(gpu))
    maml = l2l.algorithms.MAML(model, lr=fast_lr)
    optimizer = optim.Adam(maml.parameters(), meta_lr)
    loss_fcn = nn.BCELoss()

    # Outer Loop (Meta-Training Loop)
    for e in range(episodes):

        loss_per_episode = 0

        Dataset.sample_tasks()

        # Inner Loop (Task-Specific Loop)
        for t, (support, query) in enumerate(Dataset):  # 10 tasks

            # Support Set Adaptation
            task_model = maml.clone()
            loss_support = 0
            Feature.load_ESIs(support)
            for j, (e3_E, e3_A, sub_E_list, label) in enumerate(Feature):

                    i_map = []

                    for sub_idx in range(len(sub_E_list)):
                        sub_E = sub_E_list[sub_idx]

                        im = task_model(e3_E.to(device=try_gpu(gpu)), e3_A.to(device=try_gpu(gpu)), sub_E.to(device=try_gpu(gpu)))
                        i_map.append(im)

                    i_map = torch.cat(i_map, 1)
                    p_hat = torch.max(i_map)
                    i_map_avg = torch.mean(i_map)

                    p_loss = loss_fcn(p_hat, label.to(device=try_gpu(gpu)))

                    loss_support += p_loss


            loss_support = loss_support/support.shape[0]
            task_model.adapt(loss_support)

            # Query Set Evaluation
            loss_query = 0
            Feature.load_ESIs(query)
            for j, (e3_E, e3_A, sub_E_list, label) in enumerate(Feature):

                i_map = []

                for sub_idx in range(len(sub_E_list)):
                    sub_E = sub_E_list[sub_idx]

                    im = task_model(e3_E.to(device=try_gpu(gpu)), e3_A.to(device=try_gpu(gpu)),
                               sub_E.to(device=try_gpu(gpu)))
                    i_map.append(im)

                i_map = torch.cat(i_map, 1)
                p_hat = torch.max(i_map)
                i_map_avg = torch.mean(i_map)

                p_loss = loss_fcn(p_hat, label.to(device=try_gpu(gpu)))

                loss_query += p_loss

            loss_query = loss_query / query.shape[0]

            loss_per_episode += loss_query

        # Meta-learner Update
        loss_per_episode = loss_per_episode / 10
        optimizer.zero_grad()
        loss_per_episode.backward()
        optimizer.step()

        print('Episode {:d} | P_loss {:.4f} | C_loss {:.4f}'.format(e + 1, loss_per_episode, i_map_avg))

    return model



def train_MetaESI(model, dataset, Feature, epochs, gpu, mini_batch = 4):

    # Model and Optimizer Initialization
    optimizer = optim.Adam(model.parameters(), 1e-4)
    loss_fcn = nn.BCELoss()

    # Mini-batch Configuration
    if mini_batch != -1:
        num_batch = int(math.floor(dataset.shape[0] / mini_batch))
    else:
        num_batch = 1
        mini_batch = dataset.shape[0]

    # Outer Training Loop (Epochs)
    for e in range(epochs):
        model.train()
        Feature.load_ESIs(dataset, shuffle=True)

        # Inner Training Loop (Mini-batches)
        for step in range(num_batch):
            loss_batch = 0

            for iter in range(mini_batch):
                idx = step * mini_batch + iter
                if idx >= dataset.shape[0]:
                    break

                e3_E, e3_A, sub_E_list, label = Feature[idx]

                i_map = []

                for sub_idx in range(len(sub_E_list)):
                    sub_E = sub_E_list[sub_idx]

                    im = model(e3_E.to(device=try_gpu(gpu)), e3_A.to(device=try_gpu(gpu)), sub_E.to(device=try_gpu(gpu)))
                    i_map.append(im)

                i_map = torch.cat(i_map, 1)
                p_hat = torch.max(i_map)
                i_map_avg = torch.mean(i_map)

                p_loss = loss_fcn(p_hat, label.to(device=try_gpu(gpu)))

                loss_batch += p_loss

            # Loss Calculation and Backpropagation
            loss_batch = loss_batch/mini_batch
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            print('Epoch {:d} | Step {:d} | P_loss {:.4f} | C_loss {:.4f}'.format(
    e + 1, step, loss_batch, i_map_avg))

    return model



def train_MetaESI_rare(Dataset_train, rare_e3, head_e3s, kshot, model, Feature, epochs, gpu, mini_batch = 4):

    # Support Set Initialization
    support_list = []
    for head_e3 in head_e3s:
        generating_samples = Dataset_train.sample_kshot_ESIs(head_e3, kshot, shuffle = True)
        generating_samples = np.where(generating_samples == head_e3, rare_e3, generating_samples)
        support_list.append(generating_samples)
    dataset = np.vstack(support_list)

    # Model and Optimizer Initialization
    optimizer = optim.Adam(model.parameters(), 1e-4)
    loss_fcn = nn.BCELoss()

    # Mini-batch Configuration
    if mini_batch != -1:
        num_batch = int(math.floor(dataset.shape[0] / mini_batch))
    else:
        num_batch = 1
        mini_batch = dataset.shape[0]

    # Outer Training Loop (Epochs)
    for e in range(epochs):

        # Dynamic Support Set Update
        support_list = []
        for head_e3 in head_e3s:
            support_list.append(Dataset_train.sample_kshot_ESIs(head_e3, kshot, shuffle=True))
        dataset = np.vstack(support_list)

        model.train()
        Feature.load_ESIs(dataset, shuffle=True)

        # Inner Training Loop (Mini-batches)
        for step in range(num_batch):
            loss_batch = 0

            for iter in range(mini_batch):
                idx = step * mini_batch + iter
                if idx >= dataset.shape[0]:
                    break

                e3_E, e3_A, sub_E_list, label = Feature[idx]

                i_map = []

                for sub_idx in range(len(sub_E_list)):
                    sub_E = sub_E_list[sub_idx]

                    im = model(e3_E.to(device=try_gpu(gpu)), e3_A.to(device=try_gpu(gpu)), sub_E.to(device=try_gpu(gpu)))
                    i_map.append(im)

                i_map = torch.cat(i_map, 1)
                p_hat = torch.max(i_map)
                i_map_avg = torch.mean(i_map)

                p_loss = loss_fcn(p_hat, label.to(device=try_gpu(gpu)))

                loss_batch += p_loss

            # Loss Calculation and Backpropagation
            loss_batch = loss_batch/mini_batch
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            print('Epoch {:d} | Step {:d} | P_loss {:.4f} | C_loss {:.4f}'.format(
                e + 1, step, loss_batch, i_map_avg))

    return model



def train_MetaESI_woMeta(dataset, Feature, epochs, gpu=0):

    PanESI = MetaESI(1280).to(device=try_gpu(gpu))
    PanESI = train_MetaESI(PanESI, dataset, Feature, epochs, gpu)

    return PanESI



def eval_MetaESI(model, dataset, Feature, gpu):

    Feature.load_ESIs(dataset, shuffle = False)
    logits = []

    # Model Evaluation and Prediction
    for j, (e3_E, e3_A, sub_E_list, label) in enumerate(Feature):
        model.eval()
        i_map = []

        for sub_idx in range(len(sub_E_list)):
            sub_E = sub_E_list[sub_idx]

            with torch.no_grad():
                im = model(e3_E.to(device=try_gpu(gpu)), e3_A.to(device=try_gpu(gpu)), sub_E.to(device=try_gpu(gpu)))
            i_map.append(im)

        i_map = torch.cat(i_map, 1)
        p_hat = torch.max(i_map)

        logits.append(float(p_hat))

    logits = np.array(logits)
    logits = logits[:, np.newaxis]
    logits = np.concatenate([dataset, logits], axis = 1)

    return logits



def fetch_e3_specific_model(e3, Dataset, Feature, gpu = 0,
                            meta_model_path = "../models/meta_model/",
                            e3_model_path = "../models/e3_specific_model/",
                            meta_model_name = "MetaESI.pth"):
    # Check whether the E3-specific learner exists in the e3_model_path directory; if it does, directly use it; otherwise, fine-tune based on the meta-model and save it to the directory before using it

    e3_model = os.listdir(e3_model_path)

    if e3 + '.pth' in e3_model:
        # Load the existing E3-specific learner
        model = MetaESI(1280).to(device=try_gpu(gpu))
        model_dict = torch.load(e3_model_path + e3 + '.pth', map_location=try_gpu(gpu))
        model.load_state_dict(model_dict)

    else:
        # Fine-tune and save the E3-specific learner
        model = MetaESI(1280).to(device=try_gpu(gpu))
        model_dict = torch.load(meta_model_path + meta_model_name, map_location=try_gpu(gpu))
        model.load_state_dict(model_dict)

        dataset_train_e3 = Dataset.sample_specific_ESIs(e3, "train")
        np.random.shuffle(dataset_train_e3)
        model = train_MetaESI(copy.deepcopy(model), dataset_train_e3, Feature, epochs = 18, gpu=gpu, mini_batch=-1)

        torch.save(MetaESI.state_dict(), e3_model_path + e3 + '.pth')

    return model


