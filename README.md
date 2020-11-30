# Seq2Emo: A Sequence to Multi-Label Emotion Classification

This repo presents the script to replicate the results of Seq2Emo, CC, BR, BR-att as described in the paper.


## Requirements
The dependent packages are listed in the `requirements.txt`. Note that a cuda device is required.

We have included the data and preprocessing script under `data` folder.

We use [GloVe 840B 300d](https://nlp.stanford.edu/projects/glove/) pretrain embedding, you can extract and put the txt file under ``data`` folder.
or set the python argument `--glove_path` to point at it.  
You also need to use the argument  `--download_elmo` to download the ELMo embedding for first time of running the code.

## Training/evaluation 
For Seq2Emo, you can get the classification result of SemEval'18 dataset by the following script.   

```
python trainer_lstm_seq2emo.py --dataset sem18 --batch_size 16 --glove_path data/glove.840B.300d.txt --download_emo --seed 0 
```
Note that the results reported in the paper are based 5 runs with `--seed` set to `1`, `2`, `3`, `4`, `5` respectively.

To change the dataset to GoEmotions, you can specify the `--dataset` option as `goemotions`. In the paper, we set the batch size to 32 for GoEmotions dataset to accelerate training. 

Similarly, you can generate the results for CC by the following:
```
python trainer_lstm_cc.py --dataset sem18 --batch_size 16 
```

For BR, there are two variants: with self-attention and w/o self-attention. You can specify the model by setting the 
`--attention` option to `self` or `None`.  
```
python trainer_lstm_binary.py --dataset sem18 --batch_size 16 --attention [self|None] 
```


