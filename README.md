 HRESCAL: Encoder
 ===============
 This work is composed of two component, namely encoder and decoder. The encoder is mainly a procedure of feature gather by Hamming distance. We first introduce it to KGC task. The decoder is the famous facotrization model RESCAL, this is why our model named HRSCAL. However, our model not report the metrics of Hit@N, MR and MRR.  We report the metric of AUC. The results are as follows. 

Experiemnt 1 :kinship, nations and UMLS
 ----------------------------------
 | Model | Kinships  | Nations | UMLS |
 |---------|--------|--------|--------|
|CP |0.9400| 0.8300| 0.9500|
 |IRM |0.6600| 0.7500| 0.7000|
|BCTF | 0.9000 |N/A |0.9800|
|RESCAL | 0.9500| 0.8400| 0.9800|
|Linear+Reg | 0.9399| - |0.8822|
|Quad+Reg | 0.9389| - |0.8811|
|Linear+Constraint | 0.9287| - |0.8018|
|Quad+Constraint |0.9384 |- |0.9107|
|HRESCAL2(ours)| 0.9991 |0.9782| 0.9923|
|HRESCAL3(ours)| 0.9986| 0.9604| 0.9997|
|HRESCAL3(ours)| 0.9986| 0.9604| 0.9997|
|Improved2| 5.25%| 16.45%| 1.26%|
|Improved3 |5.12% |14.33%| 2.01%|

Experiemnt 2: Countries S1, S2 and S3
-----------------------
 |Model|Countries S1|Countries S2|Countries S3|
 |---------|--------|--------|--------|
HolE | 0.9970| 0.7720 |0.6970|
ComplEx |0.9977| 0.9075 |0.5474|
NTPλ| 1.0000| 0.9304 |0.7726|
MINERVA| 1.0000| 0.9304| 0.7726|
NeuralLP| 1.0000| 0.7510| 0.9220|
GNTP|  1.0000| 0.9348| 0.9127|
HRESCAL2(ours)| 0.9990| 0.9987 |0.9982|
HRESCAL3(ours) |0.9963| 0.9940| 0.9909|
Improved2 |-0.10% |7.34%| 8.57%|
Improved3 |-0.37%| 6.84% |8.57%|

Experiemnt 3:  FB15K FB15K237
------------------------
|Model |FB15K |FB15K237|
|----------------|-----------------|----------------|
|RuleN|  - |0.9225|
|GNN|  - |0.9337|
|RESCAL| - |0.9761|
|Non Neg RESCAL|  - |0.9781|
|TransE |- |0.5084|
|DisMult| - |0.7028|
|ComplEx|  - |0.6764|
|Linear+Reg|  -| 0.9649|
|Quad+Reg|  - |0.9720|
|Linear+Constraint|  - |0.8000|
|Quad+Constraint|  - |0.9459|
|HRESCAL2(ours)| 0.9907| 0.9722|
|Improved2| - |-0.61%|

Experiemnt 4: Time improvement
----------------
|Dataset| Model| Rank 10 |Rank 20| Rank 40|
|---------|--------|--------|--------|------------|
|Kinships |RESCAL| 10.76| 11.06| 12.08|
||HRESCAL2| 2.38| 2.45 |2.51|
|Nations| RESCAL| 27.39| 27.33| 27.83|
||HRESCAL2 |3.46 |3.63 |3.77|
|UMLS |RESCAL| 28.40| 29.15| 29.96|
||HRESCAL2 |4.84 |5.08| 5.24|

 HRESCAL: decoder
 ================
Our decoder is RESCAL, this part of code is from the [RESCAL](https://github.com/mnick/rescal.py), the details of this model, please refer to the link.

Usage
=====
When we use the model HRESCAL, we should first convert the data into .mat, then change the path to your file on your equipment, then your can run it.

Convert data format
-------------------
You can use the following command:
```python
python yourpath/getMat_txt.py
```
or
```python
python yourpath/getMat_tsv.py
```
We just provide the txt  and tsv convertor. Other convertors are similar.

Run the example
---------------
```python
python yourpath/KB_kinships_best.py
```
We provide all of our running code, you can run them at your will.

License
-------
rescal.py is licensed under the GPLv3 <http://www.gnu.org/licenses/gpl-3.0.txt>
