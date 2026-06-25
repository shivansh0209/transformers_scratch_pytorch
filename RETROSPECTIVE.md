# Learnings

1. many a times I get confused that if I need a transformation of matrrix then should I do it attain it by cross multiplication with another matrix or by a linear layer like both of them do the same work but just I am confsed many a times. SO if the params need to be learned during training then use layer

2. Got to know two ways of multi head attention:
* Expanding then restoring original shape
* contracting and then restoring the original shape

3. Also got to know that pytorch has very much limitations like it cannot track normal python lists, it needs its own, similarly we have nn.Parameteres, nnModuleList, nn.ModuleDict. Also pyTorch has dedicated functions for AI world like torch.tril which return a lower triangular matrix

4. Misconception: Also I thought that cross attention will have only one head but it also has multi head attention. Clarity: Every dimension has its own gamma and beta values which is common for all tokens

## Optimization tips
* Always use pyTorch things as some mentioned above, even if using basic maths use pyTorch functions and overloaded operators. Like one very powerful is that when you add two tensors (1, a, b) and (x, a, b), after noticing that dimensions a and b perfectly matches it automatically duplicated the value of a,b while adding WITHOUT ALLOCATING ANY EXTRA MEMORY.

* .view(shape) changes how the structural representation(shape) should change without using any extra memory. Many PyTorch geometry-changing operations do not actually move data around in memory. Instead, they simply change the metadata (the strides and shape) to create a new "view" of the exact same memory block. Common operations that create non-contiguous tensors include:.t() or .transpose().permute().narrow() or tensor slicingWhen a tensor becomes non-contiguous, its elements are no longer ordered sequentially in RAM. This causes certain downstream operations (most notably .view()) to throw a runtime error because they require memory to be sequential for fast processing. Hence we use contiguous