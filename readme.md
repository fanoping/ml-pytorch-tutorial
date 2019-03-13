# PyTorch tutorial

**TA tutorial,  Machine Learning (2019 Spring)**

## Contents
* Package Requirements
* NumPy Array Manipulation
* PyTorch
* Start building a model

## Package Requirement
**Note: This is a tutorial for `PyTorch==1.0.1` version**
* PyTorch == 1.0.1
* NumPy >= 1.14
* SciPy == 1.2.1

## NumPy Array Manipulation 
Some useful functions that you may use for managing your training data. We **must** carefully check our data dimensions are logically correct.

* `np.concatenate((arr_1, arr_2, ...), axis=0)`
  
   Note that the shape of array in the sequence should be the same except the dimension corresponds to the axis.
   
   ```
       # concatenate two array
       a1 = np.array([[1, 2], [3, 4], [5, 6]])    // shape: (3, 2)
       a2 = np.array([[3, 4], [5, 6], [7, 8]])    // shape: (3, 2)

       # along the axis = 0
       a3 = np.concatenate((a1, a2), axis=0)      // shape: (6, 4)
   
       # along the axis = 1
       a4 = np.concatenate((a1, a2), axis=1)      // shape: (3, 4)
   ```
   
* `np.transpose(arr, axis)`
  
   Mostly we use it to align the dimension of our data.
   ```
       # transpose 2D array
       a5 = np.array([[1, 2], [3, 4], [5, 6]])    // shape: (3, 2)
       np.transpose(a5)                           // shape: (2, 3)
   ```
   
   We can also permute multiple axis of the array.
   
   ```
       a6 = np.array([[[1, 2], [3, 4], [5, 6]]])  // shape: (1, 3, 2)
       np.transpose((a6), axes=(2, 1, 0))         // shape: (2, 3, 1)
   ```
   
## PyTorch

### Tensor Manipulation

A `torch.tensor` is conceptually identical to a numpy array, but with GPU support and additional attributes to allow Pytorch operations. 

* Create a tensor

    ```python
        b1 = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    ```

* Some frequently-used functions you can use
    ```python
        b1.size()               # to check to size of the tensor
        						# torch.Size([1, 2, 3])
        b1.view((1, 3, 2))      # same as reshape in numpy
        						# tensor([[[1, 2],
             					#		   [3, 4],
             					#		   [5, 6]]])
        b1.squeeze()    		# removes all the dimensions of size 1 
        						# tensor([[1, 2, 3],
            					#		  [4, 5, 6]])
        b1.unsqueeze()      	# inserts a new dimension of size one in a specific position
        						# tensor([[[[1, 2, 3],
              					#			[4, 5, 6]]]])
    ```

* Other manipulation functions are similar to that of NumPy, we omitted it here for simplification. For more information, please check the PyTorch documentation: https://pytorch.org/docs/stable/tensors.html

###Tensor Attributes

- Some important attributes of `torch.tensor`

- ```python
      b1.grad					# gradient of the tensor
      b1.grad_fn				# the gradient function the tensor
      b1.is_leaf				# check if tensor is a leaf node of the graph
      b1.requires_grad		# if set to True, starts tracking all operations performed
  ```

### Autograd

**torch.Auotgrad** is a package that provides functions implementing differentiation for scalar outputs.

For example:
* Create a tensor and set `requires_grad=True` to track the computation with it.

    ```python
        x1 = torch.tensor([[1., 2.],
                           [3., 4.]], requires_grad=True)
      # x1.grad 			None 
      # x1.grad_fn 			None
      # x1.is_leaf 			True
      # x1.requires_grad	True
        
        x2 = torch.tensor([[1., 2.],
                           [3., 4.]], requires_grad=True)
      # x2.grad 			None 
      # x2.grad_fn 			None
      # x2.is_leaf 			True
      # x2.requires_grad	True
    ```

    It also enables the tensor to do gradient computations later on.

    Note: Only floating dtype can require gradients.

* Do some simple operation

    ```python
        z = (0.5 * x1 + x2).sum()
      # x2.grad 			None 
      # x2.grad_fn 			<SumBackward0>
      # x2.is_leaf 			False
      # x2.requires_grad	True
    ```

* Call `backward()` function to compute gradients automatically
  
    ```python
        z.backward()	# this is identical to calling z.backward(torch.tensor(1.))
    ```

* Check the gradients using `.grad`
  
    ```
        x1.grad
        x2.grad
    ```
    
    Output will be something like this
    
    ```
        tensor([[[0.5000, 0.5000],        // x1.grad
                 [0.5000, 0.5000]]])
        tensor([[[1., 1.],                // x2.grad
                 [1., 1.]]])
    ```

## Start building a model
See example: [link](https://github.com/fanoping/ml-pytorch-tutorial/blob/master/mnist_pytorch.ipynb)