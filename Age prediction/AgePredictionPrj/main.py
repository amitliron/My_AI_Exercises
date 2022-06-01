from matplotlib                          import pyplot
from matplotlib.image                    import imread
from torchvision.datasets                import ImageFolder
from torch.optim.lr_scheduler            import OneCycleLR
from sklearn.metrics                     import r2_score


import numpy                  as np
import matplotlib.pyplot      as plt
import matplotlib.image       as mpimg
import torchvision.transforms as transforms
import torch.nn               as nn
import torch.nn.functional    as F
import torch.optim            as optim
import torchinfo


import torch
import os
import random
import matplotlib
import torchvision
import torchinfo
import gc
import os
import time


TRAIN_FOLDER = "/home/amitli/Datasets/Age prediction/20-50/train"
TEST_FOLDER  = "/home/amitli/Datasets/Age prediction/20-50/test"


 #-- ImageNet statistics:
vMean = np.array([0.48501961, 0.45795686, 0.40760392])
vStd  = np.array([0.22899216, 0.224     , 0.225     ])

oTransforms = transforms.Compose([
    #transforms.Resize    (224),
    #transforms.CenterCrop(224),
    transforms.ToTensor  (),
    transforms.Normalize (mean=vMean, std=vStd),
])


batchSize            = 32
oDataSet             = ImageFolder(root=TRAIN_FOLDER, transform=oTransforms)
oTrainSet, oTestSet  = torch.utils.data.random_split(oDataSet, np.round([0.9 * len(oDataSet), 0.1 * len(oDataSet)]).astype(int))

oTrainSet.transform  = oTransforms
oTestSet .transform  = oTransforms

oTrainDL  = torch.utils.data.DataLoader(oTrainSet,   batch_size=batchSize, num_workers=2, persistent_workers=True)
oTestDL   = torch.utils.data.DataLoader(oTestSet,    batch_size=batchSize, num_workers=2, persistent_workers=True)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def GetPretrainedModel():
    oModel = torchvision.models.resnet50(pretrained=True)
    # -- freeze weights:
    for mParam in oModel.parameters():
        if False == isinstance(mParam, nn.BatchNorm2d):
            mParam.requires_grad = False

    # -- Replace classifier head:
    dIn = oModel.fc.in_features
    oModel.fc = nn.Sequential(
        nn.Linear(dIn, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        # nn.Linear(128, 31)
        nn.Linear(128, 1)
    )

    return oModel


torchinfo.summary(GetPretrainedModel(), (32, 3, 128, 128))


def Epoch(oModel, oDataDL, Loss, Metric, oOptim=None, oScheduler=None, bTrain=True):
    epochLoss = 0
    epochMetric = 0
    count = 0
    nIter = len(oDataDL)
    vLR = np.full(nIter, np.nan)
    DEVICE = next(oModel.parameters()).device  # -- CPU\GPU

    oModel.train(bTrain)  # -- train or test

    # -- Iterate over the mini-batches:
    for ii, (mX, vY) in enumerate(oDataDL):
        # -- Move to device (CPU\GPU):
        mX = mX.to(DEVICE)
        vY = vY.to(DEVICE)

        # -- Forward:
        if bTrain == True:
            # -- Store computational graph:
            mZ = oModel(mX)
            loss = Loss(mZ, vY)
        else:
            with torch.no_grad():
                # -- Do not store computational graph:
                mZ = oModel(mX)
                loss = Loss(mZ, vY)

        # -- Backward:
        if bTrain == True:
            oOptim.zero_grad()  # -- set gradients to zeros
            loss.backward()  # -- backward
            oOptim.step()  # -- update parameters
            if oScheduler is not None:
                vLR[ii] = oScheduler.get_last_lr()[0]
                oScheduler.step()  # -- update learning rate

        Nb = vY.shape[0]
        count += Nb
        epochLoss += Nb * loss.item()
        epochMetric += Nb * Metric(mZ, vY)
        print(f'\r{"Train" if bTrain else "Val"} - Iteration: {ii:3d} ({nIter}): loss = {loss:2.6f}', end='')

    print('', end='\r')
    epochLoss /= count
    epochMetric /= count

    return epochLoss, epochMetric, vLR


def TrainModel(oModel, oTrainData, oValData, Loss, Metric, nEpochs, oOptim, oScheduler=None, Epoch=Epoch, sModelName='BestParams'):

    vTrainLoss   = np.full(nEpochs, np.nan)
    vTrainMetric = np.full(nEpochs, np.nan)
    vValLoss     = np.full(nEpochs, np.nan)
    vValMetric   = np.full(nEpochs, np.nan)
    vLR          = np.full(0,       np.nan)
    bestMetric   = -float('inf')

    for epoch in range(nEpochs):
        startTime                    = time.time()
        trainLoss, trainMetric, vLRi = Epoch(oModel, oTrainData, Loss, Metric, oOptim, oScheduler, bTrain=True ) #-- train
        valLoss,   valMetric,   _    = Epoch(oModel, oValData,   Loss, Metric,                     bTrain=False) #-- validate
        epochTime                    = time.time() - startTime

        #-- Display:
        if epoch % 10 == 0:
            print('-' * 120)
        print('Epoch '            f'{epoch       :03d}:',   end='')
        print(' | Train loss: '   f'{trainLoss   :6.3f}',   end='')
        print(' | Val loss: '     f'{valLoss     :6.3f}',   end='')
        print(' | Train Metric: ' f'{trainMetric :6.3f}',   end='')
        print(' | Val Metric: '   f'{valMetric   :6.3f}',   end='')
        print(' | epoch time: '   f'{epochTime   :6.3f} |', end='')

        vTrainLoss  [epoch] = trainLoss
        vTrainMetric[epoch] = trainMetric
        vValLoss    [epoch] = valLoss
        vValMetric  [epoch] = valMetric
        vLR                 = np.concatenate([vLR, vLRi])

        #-- Save best model (early stopping):
        if valMetric > bestMetric:
            bestMetric = valMetric
            try   : torch.save(oModel.state_dict(), sModelName + '.pt')
            except: pass
            print(' <-- Checkpoint!')
        else:
            print('')

    #-- Load best model (early stopping):
    oModel.load_state_dict(torch.load(sModelName + '.pt'))

    return vTrainLoss, vTrainMetric, vValLoss, vValMetric, vLR



def R2Score(vHatY, vY):
    vY    = vY   .detach().cpu().view(-1)
    vHatY = vHatY.detach().cpu().view(-1)
    return r2_score(vY, vHatY)



nEpochs    = 30
nIter      = nEpochs * len(oTrainDL)
Loss       = nn.MSELoss  ()
Metric     = R2Score

oModel     = GetPretrainedModel     ().to(DEVICE)
oOptim     = optim.AdamW            (oModel.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=2e-4)
oScheduler = OneCycleLR             (oOptim, max_lr=2e-2, total_steps=nIter)


lHistory   = TrainModel(oModel, oTrainDL, oTestDL, Loss, Metric, nEpochs, oOptim, oScheduler, Epoch=Epoch, sModelName='AE')