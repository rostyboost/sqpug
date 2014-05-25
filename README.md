## SqPUG
My own simple large-scale ML lib to play with D.

For now, implements http://arxiv.org/pdf/1309.2375v2.pdf.

```
benoit sqpug (master):$ dmd -release -inline -O src/sqpug.d src/Common.d src/Hasher.d src/IO.d src/Learner.d src/PredictionServer.d src/Serializer.d
benoit sqpug (master):$ time ./sqpug --data test_data/rcv1/train.txt --lambda 0.0001 --bits 18 --loss logistic --passes 5 --test test_data/rcv1/test.txt
Learn model on test_data/rcv1/train.txt
Starting learning on 781265 datapoints.
Duality gap: 0.0890009
Duality gap: 0.0616458
Duality gap: 0.0430221
Duality gap: 0.0304152
Duality gap: 0.0212375
Duality gap: 0.0146649
Duality gap: 0.0102986
Duality gap: 0.00707041
Duality gap: 0.00476217
Duality gap: 0.00325878
Duality gap: 0.00222923
Stopped SDCA after 3906325 sampled points.
Evaluate model on test_data/rcv1/test.txt
Average pred: 0.465213
Average label: 0.465938
10786 positives, 12363 negatives.
Total error: 0.0713638
Baseline error: 0.501793

real	0m22.975s
user	0m29.828s
sys	0m0.625s
```
