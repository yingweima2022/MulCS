
## ablation study about Monolingual and Multilingual

### Experiment Description

We train and evaluate our method and NBoW in two ways. One is to mix all multilingual data together for training (the method in the paper), which can effectively utilize more data. The other is to use multilingual mixed data that maintains the same scale as monolingual data. The experimental results are shown in the following table.

| Model | C | Java | Python | Avg |
| :- | :- | :- | :- | :- |
| NBoW |0.713 | 0.612 |0.605 | 0.643 |
| MuLNBoW+ | 0.734(+2.9) | 0.616(+0.7) | 0.612(+1.2) | 0.654(+1.7) |
| MuLNBoW-|0.622(-14.6) |0.502(-21.9) | 0.435(-39.1)| 0.520(-23.7)|
| SingleMulCS |0.721 |  0.619 | 0.634 | 0.658|
| MulCS+ |0.786(+9.0)| 0.667(+7.8)| 0.719(+13.4)| 0.724 (+10.0)|
| MulCS- |0.651(-10.8) | 0.517(-19.7) |0.472(-34.3) | 0.547 (-20.3)|


Among them, MuLNBoW+ and MulCS+ are the results of training using the first way, and MulNBoW- and MulCS- are the results of training using the second way. Through observation, we can draw the following conclusions:
* Both MuLNBoW+ and MulCS+ improve the single-language search performance, and MulCS+ improves by 10.0%, indicating that our method can make better use of multilingual data and improve search performance.
* Both MuLNBoW- and MulCS- degrade monolingual search performance, indicating that using more monolingual data is better than using multilingual data of the same size. But MulCS- degrades less performance than MuLNBoW-, showing that our method can better utilize limited multilingual data.
In conclusion, our method can better unify representations between multiple languages and thus make better use of multilingual data.
