# Self-training using Self-supervised Learning

## Self-supervised Combined training
`SS_train.py` exposes two extra parameters: `lambda_u` default set to 1 and for combined training self-training type
`st_type` set to "combined".
Logs will be saved at `ss-{1}_logs/dataset/network/*{2}*` {1}=loss type, {2}= combined


## Self-supervised Pretraining

`SS_pretrain.py` exposes one extra parameter for number of pretraining epochs `pretraining_epochs` default set to 200.
Logs will be saved at `ss-pretrain-{}_logs/dataset/network/` {}=loss type


## Single-step Combined training
`SS_train.py` exposes two extra parameters: `lambda_u` default set to 1 and for single-step combined training of 
self-supervised training,   self-training type
`st_type` set to "stssc".

Logs will be saved at `ss-{1}_logs/dataset/network/*{2}*` {1}=loss type, {2}= stssc

## Citation information
```
@misc{sahito2021better,
      title={Better Self-training for Image Classification through Self-supervision}, 
      author={Attaullah Sahito and Eibe Frank and Bernhard Pfahringer},
      year={2021},
      eprint={2109.00778},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
