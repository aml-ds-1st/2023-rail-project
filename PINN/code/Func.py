import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import math
from sklearn.metrics import r2_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def Train(model, model_type, train_DL, val_DL, criterion, optimizer, 
          EPOCH, BATCH_SIZE, TRAIN_RATIO,
          save_model_path, save_history_path, d_max=None, d_min=None, **kwargs):

    if "LR_STEP" in kwargs:
        scheduler = StepLR(optimizer, step_size = kwargs["LR_STEP"], gamma = kwargs["LR_GAMMA"])
    else:
        scheduler = None

    loss_history = {"train":[], "val":[]}
    best_loss = 9999
    for ep in range(EPOCH): 
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {ep+1}, current_LR = {current_lr}")

        model.train() # train mode로 전환
        train_loss = loss_epoch(model, model_type, train_DL, criterion, optimizer, d_max, d_min)
        loss_history["train"] += [train_loss]

        model.eval() # test mode로 전환
        with torch.no_grad():
            val_loss = loss_epoch(model, model_type, val_DL, criterion, d_max=d_max, d_min=d_min)
            loss_history["val"] += [val_loss]

            if val_loss < best_loss: # early stopping
                best_loss = val_loss
                # optimizer도 같이 save하면 여기서부터 재학습 시작 가능
                torch.save({"model":model,
                            "ep":ep,
                            "optimizer":optimizer,
                            "scheduler":scheduler}, save_model_path)
        if "LR_STEP" in kwargs:
            scheduler.step()
        # print loss
        torch.set_printoptions(precision=2)
        print(f"train loss: {train_loss:.8f}, "
              f"val loss: {val_loss:.8f}, "
              f"time: {round(time.time()-epoch_start)} s")
        print("-"*20)

   
    torch.save({"loss_history": loss_history,
                "EPOCH": EPOCH,
                "BATCH_SIZE": BATCH_SIZE,
                "TRAIN_RATIO": TRAIN_RATIO}, save_history_path)
    
    return loss_history

def Test(model, model_type, test_DL, criterion, d_max=None, d_min=None):
    model.eval()
    with torch.no_grad():
        test_loss = loss_epoch(model, model_type, test_DL, criterion, d_max, d_min)
    print()
    print(f"Test loss: {test_loss:.4f}")
    return test_loss

def loss_epoch(model, model_type, DL, criterion, optimizer = None, d_max=None , d_min=None):
    N = len(DL.dataset) # the number of data
    rloss = 0; rcorrect = 0; avg_cost=0; y_init=torch.tensor([[0.5]], requires_grad=True).to(DEVICE)
    for x_batch, y_batch in tqdm(DL, leave=False):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        # inference
        y_hat = model(x_batch)
        # loss
        if model_type=="PINN":
            Gloss, y_init= gov_loss2(x_batch ,y_batch, y_hat, y_init, d_max, d_min)
            loss = criterion(y_hat, y_batch)+ Gloss
        elif model_type=="Only_PINN":
            Gloss, y_init= gov_loss2(x_batch ,y_batch, y_hat, y_init, d_max, d_min)
            loss = Gloss
        else:
            loss = criterion(y_hat, y_batch)
        # update
        if optimizer is not None:
            optimizer.zero_grad() # gradient 누적을 막기 위한 초기화
            loss.backward() # backpropagation
            optimizer.step() # weight update
        # loss accumulation
        avg_cost+=loss/x_batch.shape[0]
    loss_e=avg_cost/N

    return loss_e

def gov_loss(x_batch , y_batch, y_hat, y_init, d_max, d_min):
    h_conv=5;   c=444;      m=6;        epsilon=0.8
    A_s=0.072;  sigma= 5.67*10**-8;     del_t=1/6
       
    y_last= y_batch[-1].unsqueeze(dim=1).to(DEVICE)
    y_batch= torch.cat((y_init,y_batch),0)[:-1].to(DEVICE)
    
    x_real=(d_min[0]+(d_max[0]-d_min[0])*x_batch[:,0]) # PINN인 경우, 스케일링을 위해 d_max와 d_min을 지정했어야함
    y_real=(d_min[12]+(d_max[12]-d_min[12])*y_batch[:,0])
    y_hat=(d_min[12]+(d_max[12]-d_min[12])*y_hat[:,0])

    Q1=(epsilon * sigma * A_s * ((y_hat+273.15) ** 4 - (x_real +273.15)** 4) + h_conv * A_s * (y_hat- x_real)) * del_t 
    Q2=c * m * (y_hat - y_real)
    Gloss=((Q1-Q2)**2).mean()
    
    return Gloss, y_last

def gov_loss2(x_batch , y_batch, y_hat, y_init, d_max, d_min):
    h_conv=5;   c=0.444;      m=6;        epsilon=0.8
    A_s=0.036;  sigma= 5.67*10**-8;     del_t=1/6
       
    y_last= y_batch[-1].unsqueeze(dim=1).to(DEVICE)
    y_batch= torch.cat((y_init,y_batch),0)[:-1].to(DEVICE)
    
    x_real_air=(d_min[0]+(d_max[0]-d_min[0])*x_batch[:,0]) 
    x_real_solar=(d_min[4]+(d_max[4]-d_min[4])*x_batch[:,4]) 
    y_real=(d_min[12]+(d_max[12]-d_min[12])*y_batch[:,0])
    y_hat=(d_min[12]+(d_max[12]-d_min[12])*y_hat[:,0])

    Q1=(x_real_solar*A_s+ h_conv * A_s * (y_hat- x_real_air)) * del_t 
    Q2=c * m * (y_hat - y_real)
    Gloss=((Q1-Q2)**2).mean()
    
    return Gloss, y_last

def gov_loss3(x_batch , y_batch, y_hat, y_init, d_max, d_min):
    h_conv=x_batch[:,6]*0.3;   c=0.444;      m=6;        epsilon=0.8
    A_s=0.036;  sigma= 5.67*10**-8;     del_t=1/6
       
    y_last= y_batch[-1].unsqueeze(dim=1).to(DEVICE)
    y_batch= torch.cat((y_init,y_batch),0)[:-1].to(DEVICE)
    
    x_real_air=(d_min[0]+(d_max[0]-d_min[0])*x_batch[:,0]) 
    x_real_solar=(d_min[4]+(d_max[4]-d_min[4])*x_batch[:,4]) 
    y_real=(d_min[12]+(d_max[12]-d_min[12])*y_batch[:,0])
    y_hat=(d_min[12]+(d_max[12]-d_min[12])*y_hat[:,0])

    Q1=(x_real_solar*A_s+ h_conv * A_s * (y_hat- x_real_air)) * del_t 
    Q2=c * m * (y_hat - y_real)
    Gloss=((Q1-Q2)**2).mean()
    
    return Gloss, y_last

def Test_plot(model, test_DL):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)

    x_batch = x_batch.to("cpu")

    plt.figure(figsize=(8,4))
    for idx in range(6):
        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx].permute(1,2,0).squeeze(), cmap="gray")
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(f"{pred_class} ({true_class})", color = "g" if pred_class==true_class else "r")     

def im_plot(DL):
    x_batch, y_batch = next(iter(DL))
    plt.figure(figsize=(8,4))

    

    for idx in range(6):
        im = x_batch[idx]
        im = im-im.min() # for imshow clipping
        im = im/im.max() # for imshow clipping

        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(im.permute(1,2,0).squeeze())
        true_class = DL.dataset.classes[y_batch[idx]]
        plt.title(true_class, color = "g") 
    print(f"x_batch size = {x_batch.shape}")

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num