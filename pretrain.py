import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import cv2
from tifffile import imread
from dataset import MyData1
from network import Model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import deque

Unlabel_Rate = 0.01

pan_np = imread('./data/hohhot/pan.tif')
print('The shape of the original pan:', np.shape(pan_np))

ms4_np = imread('./data/hohhot/ms4.tif')
print('The shape of the original MS:', np.shape(ms4_np))

train_np = np.load("./data/hohhot/train.npy")
print('The shape of the train label', np.shape(train_np))

test_np = np.load("./data/hohhot/test.npy")
print('The shape of the test label', np.shape(test_np))

Ms4_patch_size = 16

Interpolation = cv2.BORDER_REFLECT_101

top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('The shape of the MS picture after padding', np.shape(ms4_np))

Pan_patch_size = Ms4_patch_size * 4
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('The shape of the PAN picture after padding', np.shape(pan_np))

# label_np=label_np.astype(np.uint8)
label_np = train_np.astype(np.uint8) + test_np.astype(np.uint8)
label_np = label_np - 1

label_element, element_count = np.unique(label_np, return_counts=True)
print('Class label:', label_element)
print('Number of samples in each category:', element_count)
Categories_Number = len(label_element) - 1
print('Number of categories labeled:', Categories_Number)
label_row, label_column = np.shape(label_np)


def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


unlabeled_xy = []

for row in range(label_row):  # 行
    for column in range(label_column):
        if label_np[row][column] == 255:
            unlabeled_xy.append([row, column])

unlabeled_xy = np.array(unlabeled_xy)
shuffle_array = np.arange(0, len(unlabeled_xy), 1)
np.random.shuffle(shuffle_array)
unlabeled_xy = unlabeled_xy[shuffle_array]

length_unlabel = len(unlabeled_xy)
using_length = length_unlabel * Unlabel_Rate
unlabeled_xy = unlabeled_xy[0:int(using_length)]
print("{} sets of unlabeled data are used".format(len(unlabeled_xy)))

unlabeled_xy = np.array(unlabeled_xy)
unlabeled_xy = torch.from_numpy(unlabeled_xy).type(torch.LongTensor)

ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
pan = np.expand_dims(pan, axis=0)
ms4 = np.array(ms4).transpose((2, 0, 1))

ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)

unlabeled_data = MyData1(ms4, pan, unlabeled_xy, Ms4_patch_size)


@torch.no_grad()
def update_ema(student_model, teacher_model, m):
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.mul_(m).add_(student_param, alpha=1 - m)

@torch.no_grad()
def update_p(student_p, teacher_p, m):
    teacher_p.mul_(m).add_(student_p, alpha=1 - m)

def iCloss(v_q, v_k, y_hat, conf, queue, temp=0.1):
    sim = torch.matmul(v_q, v_k.T) / temp
    sim_total = torch.exp(sim)

    y_hat = y_hat.view(-1)
    Cpos = (y_hat.unsqueeze(1) == y_hat.unsqueeze(0)).float()

    conf = conf.view(-1)
    ave_conf = conf.mean()
    conf = (conf >= ave_conf)
    mask = conf | queue
    conf = mask.float()
    Cconf = conf.unsqueeze(1) * conf.unsqueeze(0)

    pos_sim = (sim_total * Cpos * Cconf).sum(dim=-1)
    neg_sim = sim_total.sum(dim=-1)
    loss1 = -torch.log(pos_sim[mask] / neg_sim[mask])
    loss1 = loss1.mean()
    labels = torch.where(~mask)[0]
    sim = sim[~mask]
    loss2 = F.cross_entropy(sim, labels)
    loss = loss1 + loss2
    return loss


def sim_scores(v, P, tau):
    return F.softmax(torch.matmul(v, P.T) / 0.1, dim=-1)


@torch.no_grad()
def sim_scores2(v, P, c, tau):
    return F.softmax((torch.matmul(v, P.T) - c) / 0.07, dim=-1)

class PLQueue:
    def __init__(self, num_samples, queue_len, device):
        self.num_samples = num_samples
        self.queue_len = queue_len
        self.device = device

        self.queue = torch.zeros((num_samples, queue_len), dtype=torch.long, device=device)

        self.ptr = torch.zeros(num_samples, dtype=torch.long, device=device)

    def update(self, indices, pseudo_labels):
        indices = torch.as_tensor(indices, device=self.device)
        pseudo_labels = pseudo_labels.view(-1)

        pos = self.ptr[indices]

        self.queue[indices, pos] = pseudo_labels

        self.ptr[indices] = (pos + 1) % self.queue_len

    def get_queue(self):
        return self.queue

def train(Stu_model, Tea_model, optimizer, scheduler, device, epochs=35, batch_size=128):
    Stu_model.train()
    with torch.no_grad():
        Tea_model.load_state_dict(Stu_model.state_dict())

    for param in Tea_model.parameters():
        param.requires_grad = False

    data_loader = DataLoader(unlabeled_data, batch_size, num_workers=0, shuffle=True, drop_last=True)

    c1 = torch.zeros(1, 16).to(device)
    c2 = torch.zeros(1, 16).to(device)
    c3 = torch.zeros(1, 16).to(device)
    c4 = torch.zeros(1, 16).to(device)

    min_loss = 10
    num_data = len(unlabeled_data)
    queue_size = 5

    pseudo_queue1 = PLQueue(num_samples=num_data, queue_len=queue_size, device=device)
    pseudo_queue2 = PLQueue(num_samples=num_data, queue_len=queue_size, device=device)
    pseudo_queue3 = PLQueue(num_samples=num_data, queue_len=queue_size, device=device)
    pseudo_queue4 = PLQueue(num_samples=num_data, queue_len=queue_size, device=device)

    for epoch in range(epochs):
        total_num, total_loss, loss1, loss2, loss3, loss4, train_bar = 0, 0.0, 0.0, 0.0, 0.0, 0.0, tqdm(data_loader)

        for batch_idx, (ms_view1, ms_view2, pan_view1, pan_view2, index, _) in enumerate(train_bar):

            ms_view1 = ms_view1.to(device)
            ms_view2 = ms_view2.to(device)
            pan_view1 = pan_view1.to(device)
            pan_view2 = pan_view2.to(device)

            out_ms_g1, out_ms_h1, out_pan_g2, out_pan_h2 = Stu_model(ms_view1, pan_view2)
            with torch.no_grad():
                out_ms_g2, out_ms_h2, out_pan_g1, out_pan_h1 = Tea_model(ms_view2, pan_view1)

            # MS to MS
            sim11 = sim_scores(out_ms_h1, Stu_model.prototypes1, 0.1)
            with torch.no_grad():
                sim1 = sim_scores2(out_ms_h2, Tea_model.prototypes1, c1, 0.07)
                c1 = 0.996 * c1 + (1 - 0.996) * (torch.matmul(out_ms_h2, Tea_model.prototypes1.T)).mean(dim=0)

            loss_1 = F.kl_div(sim11.log(), sim1, reduction="batchmean")

            weight1, pseudo_label1 = sim1.max(dim=1, keepdim=True)
            # loss_m2m = iCloss(out_ms_h1, out_ms_h2, pseudo_label1, weight1)

            # PAN to PAN
            sim22 = sim_scores(out_pan_h2, Stu_model.prototypes2, 0.1)
            with torch.no_grad():
                sim2 = sim_scores2(out_pan_h1, Tea_model.prototypes2, c2, 0.07)
                c2 = 0.996 * c2 + (1 - 0.996) * (torch.matmul(out_pan_h1, Tea_model.prototypes2.T)).mean(dim=0)

            loss_2 = F.kl_div(sim22.log(), sim2, reduction="batchmean")

            weight2, pseudo_label2 = sim2.max(dim=1, keepdim=True)
            # loss_p2p = iCloss(out_pan_h2, out_pan_h1, pseudo_label2, weight2)

            # MS to PAN
            sim33 = sim_scores(out_ms_g1, Stu_model.prototypes3, 0.1)
            with torch.no_grad():
                sim3 = sim_scores2(out_pan_g1, Tea_model.prototypes3, c3, 0.07)
                c3 = 0.996 * c3 + (1 - 0.996) * (torch.matmul(out_pan_g1, Tea_model.prototypes3.T)).mean(dim=0)

            loss_3 = F.kl_div(sim33.log(), sim3, reduction="batchmean")

            weight3, pseudo_label3 = sim3.max(dim=1, keepdim=True)
            # loss_m2p = iCloss(out_ms_g1, out_pan_g1, pseudo_label3, weight3)

            # PAN to MS
            sim44 = sim_scores(out_pan_g2, Stu_model.prototypes3, 0.1)
            with torch.no_grad():
                sim4 = sim_scores2(out_ms_g2, Tea_model.prototypes3, c4, 0.07)
                c4 = 0.996 * c4 + (1 - 0.996) * (torch.matmul(out_ms_g2, Tea_model.prototypes3.T)).mean(dim=0)

            loss_4 = F.kl_div(sim44.log(), sim4, reduction="batchmean")

            kl_loss = loss_1 + loss_2 + loss_3 + loss_4

            weight4, pseudo_label4 = sim4.max(dim=1, keepdim=True)
            # loss_p2m = iCloss(out_pan_g2, out_ms_g2, pseudo_label4, weight4)

            pseudo_queue1.update(index, pseudo_label1)
            pseudo_queue2.update(index, pseudo_label2)
            pseudo_queue3.update(index, pseudo_label3)
            pseudo_queue4.update(index, pseudo_label4)

            if epoch > (queue_size - 1):
                q1 = pseudo_queue1.get_queue()[index]
                q2 = pseudo_queue2.get_queue()[index]
                q3 = pseudo_queue3.get_queue()[index]
                q4 = pseudo_queue4.get_queue()[index]

                queue1 = (q1 == q1[:, 0:1]).all(dim=1)
                queue2 = (q2 == q2[:, 0:1]).all(dim=1)
                queue3 = (q3 == q3[:, 0:1]).all(dim=1)
                queue4 = (q4 == q4[:, 0:1]).all(dim=1)

                loss_m2m = iCloss(out_ms_h1, out_ms_h2, pseudo_label1, weight1, queue1)
                loss_p2p = iCloss(out_pan_h2, out_pan_h1, pseudo_label2, weight2, queue2)

                intra_loss = loss_m2m + loss_p2p

                loss_m2p = iCloss(out_ms_g1, out_pan_g1, pseudo_label3, weight3, queue3)
                loss_p2m = iCloss(out_pan_g2, out_ms_g2, pseudo_label4, weight4, queue4)

                inter_loss = loss_m2p + loss_p2m

                align_loss = intra_loss + inter_loss

                loss = align_loss + kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_num += batch_size
                total_loss += loss.item() * batch_size
                loss1 += intra_loss.item() * batch_size
                loss2 += inter_loss.item() * batch_size
                loss3 += kl_loss.item() * batch_size

                train_bar.set_description(
                    'Train Epoch: [{}/{}] loss: {:.4f} loss: {:.4f} loss: {:.4f} loss: {:.4f}'.format(epoch, epochs,
                                                                                                      total_loss / total_num,
                                                                                                      loss1 / total_num,
                                                                                                      loss2 / total_num,
                                                                                                      loss3 / total_num,
                                                                                                      ))


                with torch.no_grad():
                    update_ema(Stu_model, Tea_model, 0.996)
                    Tea_model.prototypes1.data = F.normalize(Tea_model.prototypes1.data, dim=1)
                    Tea_model.prototypes2.data = F.normalize(Tea_model.prototypes2.data, dim=1)
                    Tea_model.prototypes3.data = F.normalize(Tea_model.prototypes3.data, dim=1)
                Stu_model.prototypes1.data = F.normalize(Stu_model.prototypes1.data, dim=1)
                Stu_model.prototypes2.data = F.normalize(Stu_model.prototypes2.data, dim=1)
                Stu_model.prototypes3.data = F.normalize(Stu_model.prototypes3.data, dim=1)



        torch.save(Stu_model.state_dict(), 'models/pretrain_hohhot.pth')
                # eva_loss = total_loss / total_num
                # if eva_loss < min_loss:
                #     torch.save(Stu_model.state_dict(), 'models/pretrain_nanjing.pth')
                #     min_loss = eva_loss

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f"Training with: {device}")

Stu_model = Model().to(device)
Tea_model = Model().to(device)

optimizer = optim.Adam(Stu_model.parameters(), lr=1e-3, weight_decay=1e-6)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
train(Stu_model, Tea_model, optimizer, scheduler, device)