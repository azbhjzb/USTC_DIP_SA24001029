import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from GAN_network import Generator, Discriminator

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_image_np = tensor_to_image(inputs[i])
        target_image_np = tensor_to_image(targets[i])
        output_image_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack([input_image_np, target_image_np, output_image_np])

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/comparison_{i}.png', comparison)

def train_one_epoch(model_G,model_D, k, l, Lambda, dataloader, optimizer_G,optimizer_D, criterion, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        model_G (nn.Module): The Generator.
        model_D (nn.Module): The Discriminator.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer_G (Optimizer): Optimizer for updating model_G parameters.
        optimizer_D (Optimizer): Optimizer for updating model_G parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model_D.train()
    model_G.train()
    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)
        batch_size = image_rgb.size(0)

        for j in range(k):
            # Train the discriminator,k is the number of times the discriminator is trained per generator update
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
            fake_labels = torch.zeros(batch_size, 1,dtype=torch.float32).to(device)
            real_outputs = model_D(image_rgb)
            real_loss = criterion(real_outputs, real_labels)

            fake_images = model_G(image_semantic)
            fake_outputs = model_D(fake_images.detach()) # detach to stop gradients from flowing back to generator
            fake_loss = criterion(fake_outputs, fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            #print(f'r_loss: {real_loss.item()}, f_loss: {fake_loss.item()}')
            optimizer_D.step()      
        for j in range(l):
            # Train the generator, l is the number of times the generator is trained per discriminator update
            optimizer_G.zero_grad()
            fake_images = model_G(image_semantic)
            fake_outputs = model_D(fake_images)
        
            g_loss = nn.L1Loss()(fake_images, image_rgb) + Lambda * criterion(fake_outputs, real_labels) 
            g_loss.backward()
            optimizer_G.step()

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, model_G(image_semantic), 'train_results', epoch)
        print(f'Epoch [{epoch + 1} / {num_epochs}], Step [{i + 1}/{len(dataloader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')

def validate(model_G,model_D, Lambda, dataloader, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model_G (nn.Module): The Generator.
        model_D (nn.Module): The Discriminator.
        dataloader (DataLoader): DataLoader for the validation data.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model_G.eval()
    model_D.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)
            batch_size = image_rgb.size(0)

            # Forward pass
            outputs = model_G(image_semantic)

            # Compute loss
            real_labels = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
            loss = nn.BCELoss()(model_D(outputs), real_labels)  * Lambda + nn.L1Loss()(outputs, image_rgb)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results', epoch)
    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1} / {num_epochs}], Validation Loss: {avg_val_loss:.4f}')


def main():
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the hyperparameters
    batch_size = 100
    num_epochs = 200
    learning_rate = 0.001
    k = 2 # Number of times to train the discriminator before training the generator
    l = 4 # Number of times to train the generator per discriminator update
    Lambda = 0.1 # weight for the adversarial loss
    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Set up the models
    model_G = Generator().to(device)
    model_D = Discriminator().to(device)

    # Set up the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(model_G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(model_D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=200, gamma=0.2)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=200, gamma=0.2)

    # Train the models
    for epoch in range(num_epochs):
        train_one_epoch(model_G, model_D, k, l, Lambda, train_loader, optimizer_G, optimizer_D, criterion, device, epoch, num_epochs)
        validate(model_G, model_D, Lambda, val_loader, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler_G.step()
        scheduler_D.step()

        # Save model checkpoint every 20 epochs
        if (epoch+1) % 20 == 0:
            os.makedirs('checkpoints_G', exist_ok=True)
            os.makedirs('checkpoints_D', exist_ok=True)
            torch.save(model_G.state_dict(), f'checkpoints_G/pix2pix_model_epoch_{epoch + 1}.pth')
            torch.save(model_D.state_dict(), f'checkpoints_D/pix2pix_model_epoch_{epoch + 1}.pth')

    # Save the final model
    torch.save(model_G.state_dict(), f'checkpoints_G/pix2pix_model_epoch_{epoch + 1}.pth')
    torch.save(model_D.state_dict(), f'checkpoints_D/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
    
        