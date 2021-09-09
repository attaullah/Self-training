# Self-training using Metric learning
`train.py`
For baseline,  Loss type `lt=cross-entropy`
Logs will be saved at `{}_logs/dataset/network/` {}=loss type

` python train.py --flagfile config/stssc.cfg`

## Contrastive loss
Loss type `lt=contrastive`



## Triplet loss
Loss type `lt=triplet`


## ArcFace loss
Loss type `lt=arcface`

## Citation info

```
@inproceedings{sahito2019semi,
  title={Semi-supervised learning using Siamese networks},
  author={Sahito, Attaullah and Frank, Eibe and Pfahringer, Bernhard},
  booktitle={Australasian Joint Conference on Artificial Intelligence},
  pages={586--597},
  year={2019},
  organization={Springer}
}
```
## References
1. Hadsell, Raia, Sumit Chopra, and Yann LeCun. "Dimensionality reduction by learning an invariant mapping." 2006 IEEE 
Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06). Vol. 2. IEEE, 2006.
2. Distance metric learning for large margin nearest neighbour classification. Kilian Q Weinberger and Lawrence K Saul.
Journal of Machine Learning Research,  10(2), 2009.
3. Deng, Jiankang, et al. "Arcface: Additive angular margin loss for deep face recognition." Proceedings of the 
IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
