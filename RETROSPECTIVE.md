# Learnings

1. many a times I get confused that if I need a transformation of matrrix then should I do it attain it by cross multiplication with another matrix or by a linear layer like both of them do the same work but just I am confsed many a times. SO if the params need to be learned during training then use layer

2. Got to know two ways of multi head attention:
* Expanding then restoring original shape
* contracting and then restoring the original shape

3. Also got to know that pytorch has very much limitations like it cannot track normal python lists, it needs its own, similarly we have nn.Parameteres, nnModuleList, nn.ModuleDict. Also pyTorch has dedicated functions for AI world like torch.tril which return a lower triangular matrix

4. Misconception: Also I thought that cross attention will have only one head but it also has multi head attention. Clarity: Every dimension has its own gamma and beta values which is common for all tokens