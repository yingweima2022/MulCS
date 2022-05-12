
## Experiment Description
To combine our representation method with current popular pre-training techniques, we use the CodeBERT model to obtain vector embeddings of query and code. CodeBERT generates high-quality text and code embeddings by pre-training on the CodeSearchNet corpus with two tasks, masked language modeling and replaced token detection. To fuse our semantic map in the fine-tuning stage, we transform the semantic map into sequences via a depth-first search algorithm, which is used as additional information for code representations. Specifically, we use CodeBERT as encoders. When encoding the code, the input of the original model is the token sequence of the code, and the modified model uses the token sequence of the code and the sequence obtained from the depth-first search traversal of the semantic graph as input. The data used in the fine-tuning phase is the same as the MulCS training data. 

The experimental results are shown in the following Table.

| Model | C| Java | Python | Avg|
| :-: | :-: | :-: | :-: | :-: |
| MulCS-w/o.CL |0.706 | 0.579 |0.525 | 0.603 |
| CodeBERT | <b>0.735</b> | 0.643 | 0.679 | 0.686 |
| CodeBERT_IR | 0.731 |<b>0.656</b> | <b>0.718</b>| <b>0.702</b>|

#### Among the results, CodeBERT_FT fine-tunes the pre-trained model on our data, CodeBERT_IR adds semantic graph sequences as additional information in fine-tuning. The results show that: 
* the pre-trained model greatly improves the performance of code search; 
* adding our semantic graph representation of IR can get additional enhancement even in a simple way during the fine-tuning stage. It would indicate a promising direction of understanding source code by combining intermediate representation and pre-training techniques. We leave a deep investigation in the future work.
