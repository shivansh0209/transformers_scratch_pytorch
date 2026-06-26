# Learnings

1. many a times I get confused that if I need a transformation of matrrix then should I do it attain it by cross multiplication with another matrix or by a linear layer like both of them do the same work but just I am confsed many a times. SO if the params need to be learned during training then use layer

2. Got to know two ways of multi head attention:
* Expanding then restoring original shape
* contracting and then restoring the original shape

3. Also got to know that pytorch has very much limitations like it cannot track normal python lists, it needs its own, similarly we have nn.Parameteres, nnModuleList, nn.ModuleDict. Also pyTorch has dedicated functions for AI world like torch.tril which return a lower triangular matrix

4. Misconception: Also I thought that cross attention will have only one head but it also has multi head attention. Clarity: Every dimension has its own gamma and beta values which is common for all tokens

## Optimization Pytorch tips
* Always use pyTorch things as some mentioned above, even if using basic maths use pyTorch functions and overloaded operators. Like one very powerful is that when you add two tensors (1, a, b) and (x, a, b), after noticing that dimensions a and b perfectly matches it automatically duplicated the value of a,b while adding WITHOUT ALLOCATING ANY EXTRA MEMORY.

* .view(shape) changes how the structural representation(shape) should change without using any extra memory. Many PyTorch geometry-changing operations do not actually move data around in memory. Instead, they simply change the metadata (the strides and shape) to create a new "view" of the exact same memory block. Common operations that create non-contiguous tensors include:.t() or .transpose().permute().narrow() or tensor slicingWhen a tensor becomes non-contiguous, its elements are no longer ordered sequentially in RAM. This causes certain downstream operations (most notably .view()) to throw a runtime error because they require memory to be sequential for fast processing. Hence we use contiguous

## Transformer Inference vs Training
Actually transfromer or more precisely the decoder of the transformer produces that number of tokens only which is being given as input to the decoder. So at time of inference if KV cache is there then we give the transformer one token only at one time unless the generation limit is reached or the transformer outputs index of 'endseq'

But if KV cache is not there then we need to give all previous generated inputs then append the output of only the last generated token(decided with the best probability among all the words in the vocab) to the answer list. SO you see without KV cache we are wasting some memory also and doing more compute basically O(n^2).

What KV cache does is it stores the K and V transfromations of all the previously generated tokens so that the input of the newly generated token can query with them also. Hence now we dont need to pass all the previous tokens generated as inputs only the prev generated so that we can calculate the K and V for the newly generated and append it to the KV cache

KV cache is not needed at time of training and mask is not needed at time of inference