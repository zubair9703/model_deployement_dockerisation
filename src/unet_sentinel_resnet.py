import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeo.models import ResNet18_Weights,ResNet50_Weights
from torchgeo.trainers import ClassificationTask
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassFBetaScore,
    MulticlassJaccardIndex,
    MultilabelAccuracy,
    MultilabelFBetaScore,
)
from segmentation_models_pytorch.losses import FocalLoss, JaccardLoss




class ConvBlock(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, input_channels, num_filters):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=input_channels, out_channels=num_filters, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels=num_filters*2, num_filters=num_filters)  # *2 for concatenation
        self.conv_block1 = ConvBlock(in_channels=num_filters, num_filters=num_filters)  # *2 for concatenation

    def forward(self, x, skip_features=None):
        x = self.upconv(x)
        if skip_features is not None:
          x = torch.cat([x, skip_features], dim=1)
          x = self.conv_block(x)
        x = self.conv_block1(x)
        return x

class center(nn.Module):
  """
  This is the middle layer of the UNet which just consists of some
  """

  def __init__(self, in_channels, out_channels):
      super().__init__()
      self.bridge = nn.Sequential(
          ConvBlock(in_channels, out_channels),
          ConvBlock(out_channels, out_channels)
      )

  def forward(self, x):
      return self.bridge(x)
  

class UnetResnetSentinel2(nn.Module):
    def __init__(self, input_channels,num_classes):
        super().__init__()
        self.weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
        model = ClassificationTask(
            model='resnet50',
            weights=self.weights,
            in_channels=input_channels,
            num_classes=num_classes,
        )

        """Freeze the encoder weights"""
        for i in model.parameters():
            i.requires_grad = False

        """Input layer"""
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        """Encoder"""
        self.e1 =  list(model.children())[0].layer1
        self.e2 =  list(model.children())[0].layer2
        self.e3 =  list(model.children())[0].layer3

        """Bottleneck"""
        self.center =  list(model.children())[0].layer4

        """Decoder"""
        self.d1 = DecoderBlock(2048,1024)
        self.d2 = DecoderBlock(1024,512)
        self.d3 = DecoderBlock(512,256)
        self.d4 = DecoderBlock(256,64)
        self.d5 = DecoderBlock(64,64)

        """Final layer"""
        self.final = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
        # nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x_ = self.maxpool(x)
        e1 = self.e1(x_)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        center = self.center(e3)
        d1 = self.d1(center,e3)
        d2 = self.d2(d1,e2)
        d3 = self.d3(d2,e1)
        d4 = self.d4(d3,x)
        d5 = self.d5(d4)
        final = self.final(d5)
        # final = torch.argmax(final, dim=2)
        return final
    


class unet(pl.LightningModule):
    def __init__(self, encoder,nc,c,loss):
        super(unet,self).__init__()
        self.encoder = encoder(nc,c)
        self.loss = loss
        self.classes = c

    # def configure_losses(self) -> None:
    #     """Initialize the loss criterion.

    #     Raises:
    #         ValueError: If *loss* is invalid.
    #     """
        loss: str = self.loss
        if loss == 'ce':
            self.criterion: nn.Module = nn.CrossEntropyLoss(
            )
        elif loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss == 'jaccard':
            self.criterion = JaccardLoss(mode='multiclass')
        elif loss == 'focal':
            self.criterion = FocalLoss(mode='multiclass', normalized=True)
        else:
            raise ValueError(f"Loss type '{loss}' is not valid.")

    # def configure_metrics(self) -> None:
        metrics = MetricCollection(
            {
                'OverallAccuracy': MulticlassAccuracy(
                    num_classes=self.classes, average='micro'
                ),
                'AverageAccuracy': MulticlassAccuracy(
                    num_classes=self.classes, average='macro'
                ),
                'JaccardIndex': MulticlassJaccardIndex(
                    num_classes=self.classes
                ),
                'F1Score': MulticlassFBetaScore(
                    num_classes=self.classes, beta=1.0, average='micro'
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, image):
        mask = self.encoder(image)
        return mask
    

    def shared_step(self, batch):
        image, mask = batch

        # Ensure that image dimensions are correct
        assert image.ndim >= 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        assert mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss: torch.Tensor = self.criterion(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1) 

        return loss,pred_mask


    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x,y = batch
        loss, xhat = self.shared_step(batch)
        self.train_metrics(xhat, y)
        values = {"loss": loss}
        self.log_dict(values,prog_bar=True)
        self.log_dict(self.train_metrics,prog_bar=True)

        # if batch_idx % 100 == 0:
        #         x_sample = x[:8]  # Take the first 8 images from the batch
        #         y_sample = y[:8]  # Take the first 8 masks from the batch
        #         x_hat_sample = xhat[:8]  # Take the first 8 predicted outputs

        #         # Create a grid for the input images
        #         image_grid = torchvision.utils.make_grid(x_sample.view(-1, 3, 64, 64), nrow=4)
                
        #         # Create a grid for the true masks
        #         mask_grid = torchvision.utils.make_grid(y_sample.view(-1, 1, 64, 64), nrow=4)  # Ensure mask has a single channel

        #         # Create a grid for the predicted outputs
        #         pred_grid = torchvision.utils.make_grid(x_hat_sample.view(-1, 1, 64, 64), nrow=4)  # Assuming output is single-channel

        #         # Log images and masks to TensorBoard
        #         # self.logger.experiment.whatever_ml_flow_supports("sample_images", image_grid, self.global_step)
        #         # self.logger.experiment.whatever_ml_flow_supports("true_masks", mask_grid, self.global_step)
        #         # self.logger.experiment.whatever_ml_flow_supports("predicted_masks", pred_grid, self.global_step)
        #         mlflow.log_figure(image_grid.cpu(),"sample_images")
        #         mlflow.log_figure(mask_grid.cpu(),"true_masks")
        #         mlflow.log_figure(pred_grid.cpu(),"predicted_masks")
        return loss

    def validation_step(self, batch, batch_idx)-> None:
        x,y = batch
        loss, xhat = self.shared_step(batch)
        self.val_metrics(xhat, y)
        values = {"vall_loss": loss}
        self.log_dict(values,prog_bar=True)
        self.log_dict(self.val_metrics,prog_bar=True)

        return loss

    # def test_step(self, batch, batch_idx)-> None:
    #     # this is the test loop
    #     x = batch
    #     x_hat = self.encoder(x['image'])
    #     y = torch.nn.functional.one_hot(x['mask'], num_classes=self.classes).permute(0,3,1,2).float()
    #     loss: Tensor = self.criterion(x_hat, y)
    #     miou = self.miou(x_hat, y)
    #     values = {"test_loss": loss, "test_Iou": miou}
    #     self.log_dict(values,prog_bar=True)
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer