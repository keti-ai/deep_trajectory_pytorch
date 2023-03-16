


## References
- Oleg Sémery. (2021). Deep learning networks. GitHub. https://github.com/osmr/imgclsmob.git

## prerequisite

pip install -r requirement.txt

## Train

python train.py

## eval

python test.py 

```python
'''
'''
for i,data in enumerate(dataloader):
    label = data[0][1]
    tracklet = data[0][0]
    # gru_out_f, gru_out_b, srh_out, pur_out,cls_out = model(torch.stack([tracklet]))
    gru_out_f, gru_out_b, srh_out, pur_out = model(torch.stack([tracklet]))
    srh_target = torch.Tensor([label[i] != label[i + 1] for i in range(len(label) - 1)]).cuda()
    y_pred_f=gru_out_f.topk(5)
    gru_out_f, gru_out_b, srh_out, pur_out = model(torch.stack([tracklet]))

# searching point prediction
srh_pred = (srh_out[0].view(track_len_in - 1, 2).topk(1)[1] == 1)

# in srh_pred , True(1) value is the cleaving point 

'''

```