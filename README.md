## Hybrid Style Siamese Network

This is the code for the paper:

[Hybrid Style Siamese Network: Incorporating style loss in complimentary apparels retrieval](https://arxiv.org/abs/1912.05014)
<br>
Mayukh Bhattacharyya, Sayan Nag

Hybrid Style Siamese Network incorporates style loss into triplet loss, in order to aid in complimentary images retrieval. In the paper, it had been used for the application of complimentary apparels retrieval.

| Model | Seed 1 | Seed 2 | Seed 3 | Mean |
| --- | --- | --- | --- | --- |
| Siamese Network | 0.1226 | 0.1323 | 0.1263 | 0.1271 |
| Hybrid Style Siamese Network | 0.1251 | 0.1343 | 0.1329 | 0.1308 | 

In order to reproduce the experiments in the paper:

```
python hssn.py -s seed_value -l learning rate -e epochs -p patience -b batch_size -f cv_fold_id --hybrid
```

```--hybrid``` is the flag for running the hybrid style siamese network. Omitting it will run the same experiment with the normal siamese network. Explanation of other parameters are there in ```hssn.py```


