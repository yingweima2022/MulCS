##  MRR Example

MRR is mainly used to evaluate search algorithms, and Reciprocal Rank refers to the reciprocal of the ranking of the first correct result. 
For example, if the correct code result has the highest similarity with the query, it ranks first in the candidate code, and the score is 1; 
the correct code result ranks in the nth place, and the score is 1/n. Mean Reciprocal Rank is the average of multiple query scores. 

Suppose there are three queries, as shown in the following figure.


| Query | Results| Correct response| Rank | Reciprocal Rank|
| :-: | :-: | :-: | :-: | :-: |
| query |falsecode1, falsecode2, truecode | truecode |3 | 1/3 |
| max | min, max, mean | max | 2 | 1/2 |
| quick sort | quick sort, merge sort, bubble sort |quick sort | 1| 1|


The MRR value of this system can be calculated as: (1/3+1/2+1)/3 = 11/18=0.61.
