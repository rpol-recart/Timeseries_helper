class Inception(nn.Module)

This PyTorch module called "Inception" is inspired by the Inception module in the Inception architecture used in the GoogLeNet neural network.


Parameters:



in_channels (int): Number of input channels (input features).

n_filters (int): Number of filters per convolution layer => out_channels = 4*n_filters.

kernel_sizes (list of int): List of kernel sizes for each convolution. Each kernel size must be an odd number that meets "kernel_size % 2 != 0". This is necessary because of padding size. For correction of kernel_sizes use function "correct_sizes".

bottleneck_channels (int): Number of output channels in the bottleneck. Bottleneck will not be used if the number of in_channels is equal to 1.

activation (torch.nn.activation): Activation function for output tensor (nn.ReLU()).

return_indices (bool): Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d.


Layers and operations:


The module contains the following layers and operations:


If in_channels > 1, then it uses a bottleneck that is a 1D convolution layer with input_channels=in_channels, output_channels=bottleneck_channels, kernel_size=1, stride=1 and bias=False. Otherwise, a pass_through layer is used.


Three 1D convolution layers using the input from the bottleneck layer. Each convolution layer has in_channels=bottleneck_channels, out_channels=n_filters, kernel_size=kernel_sizes[0],1 or 2, stride=1, padding=kernel_sizes[0] // 2,1 or 2 // 2 and bias=False, respectively.


A 1D convolution layer with input from the maximum pooling layer. The layer has in_channels=in_channels, out_channels=n_filters, kernel_size=1, stride=1, padding=0 and bias=False.


Max pooling layer with kernel_size=3, stride=1, padding=1 and return_indices, if self.return_indices=True.


A batch normalization layer that normalizes the output of the concatenation of the previous four layers.


An activation function, here the Rectified Linear Unit, ReLU.


Forward method:


The module has a forward method that performs the following operations:



Pass the input through the bottleneck and the maximum pooling layer.

Perform convolution on each input of the the bottleneck layer and the maximum pooling layer.

Concatenate the outputs from the previous convolution layers.

Normalize the output using batch normalization.

Use an activation function on the normalized output.


If return_indices=True, the method will return the max-pooling indices along with the output. Otherwise, only the output is returned. 

class InceptionBlock(nn.Module)

This is a PyTorch module called InceptionBlock, consisting of three Inception modules. 


The block takes in the following arguments: 



in_channels: integer, the number of input channels 

n_filters: integer, the number of filters to use 

kernel_sizes: list of integers, the kernel sizes to use in the Inception modules 

bottleneck_channels: integer, the number of bottleneck channels to use in the Inception modules 

use_residual: boolean, whether to use residual connections or not 

activation: activation function to apply to the intermediate activations 

return_indices: boolean, whether to return the indices used for max-pooling in the intermediate Inception modules or not


For each of the three Inception modules, a separate instance is created, with the output of the first one being the input to the second, and the output of the second being the input to the third. 


If use_residual is True, a residual connection is added to the output of the Inception modules. 


Finally, the forward method takes the input tensor X, passes it through the Inception modules and returns either the output tensor of the Inception modules, or both the output tensor and the indices used for max-pooling in the intermediate Inception modules.

class InceptionTranspose(nn.Module)

This is a implementation of a neural network module called InceptionTranspose. This module contains several convolutional layers and a max pooling and an unpooling layer. The purpose of this neural network is to transpose/upsample an input tensor to a higher resolution for semantic segmentation tasks, which relies on detailed spatial information. 


The InceptionTranspose constructor takes as inputs the number of input channels, the number of output channels, a list of kernel sizes for each convolution layer, the number of output channels in the bottleneck layer, and the type of activation function to be applied on the output tensor. 


The forward method takes an input tensor X and indices (indices generated from max pooling layer) and returns the upsampled tensor.

class InceptionTransposeBlock(nn.Module):

Этот код определяет класс InceptionTransposeBlock, который является наследником класса nn.Module из PyTorch.


Конструктор класса принимает несколько параметров:



in_channels: количество входных каналов

out_channels: количество выходных каналов

kernel_sizes: размеры ядер свертки для каждой ветки Inception-модуля в блоке, заданные в виде списка

bottleneck_channels: количество каналов в узком месте веток Inception-модуля

use_residual: флаг, указывающий, следует ли использовать остаточное соединение

activation: функция активации, применяемая к выходу блока


В методе __init__ создаются три Inception-модуля для выполнения операций свертки на входных данных. Также создаются объекты use_residual и activation. Если use_residual = True, блок также включает остаточное соединение residual, которое применяет свертку и нормализацию к входным данным.


Метод forward выполняет прямой проход через блок, начиная с первого Inception-модуля, который принимает входные данные X и индекс indices[ 2]. Затем проход выполняется по другим веткам Inception-модуля, и результат суммируется с остаточным соединением, если такое присутствует.

