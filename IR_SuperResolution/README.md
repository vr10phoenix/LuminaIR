# Intro 
pipline made for specifically to achieve TIR image super-resolution

### Custom Dataloader 
Dataloader and Preprocessing updated as per given guidelines by BAH 26 

### envirionment 
Colab Notebook  
GPU : T4 gpu 

### Resuls : 

```

Epoch [01/50] | Compound Loss: 46866.077362 | Sec/Epoch: 100.1s
Epoch [02/50] | Compound Loss: 17740.692032 | Sec/Epoch: 50.3s
Epoch [03/50] | Compound Loss: 9895.448097 | Sec/Epoch: 49.5s
Epoch [04/50] | Compound Loss: 9439.604111 | Sec/Epoch: 49.8s
Epoch [05/50] | Compound Loss: 5225.085350 | Sec/Epoch: 49.6s
Epoch [06/50] | Compound Loss: 4819.505478 | Sec/Epoch: 49.8s
Epoch [07/50] | Compound Loss: 4739.990356 | Sec/Epoch: 49.9s
Epoch [08/50] | Compound Loss: 4546.458824 | Sec/Epoch: 50.0s
Epoch [09/50] | Compound Loss: 3775.344109 | Sec/Epoch: 50.1s
Epoch [10/50] | Compound Loss: 4472.978394 | Sec/Epoch: 50.0s
Epoch [11/50] | Compound Loss: 4849.237831 | Sec/Epoch: 49.9s  
.........  
Epoch [45/50] | Compound Loss: 1412.901590 | Sec/Epoch: 50.1s
Epoch [46/50] | Compound Loss: 1404.215868 | Sec/Epoch: 50.0s
Epoch [47/50] | Compound Loss: 1403.784193 | Sec/Epoch: 49.9s
Epoch [48/50] | Compound Loss: 1397.165783 | Sec/Epoch: 50.0s
Epoch [49/50] | Compound Loss: 1394.368469 | Sec/Epoch: 50.0s
Epoch [50/50] | Compound Loss: 1392.799917 | Sec/Epoch: 50.0s
 Trained parameters cleanly written to 'swinir_thermal_v1.pth'
```
#### Obtained Image
![result](https://github.com/vr10phoenix/LuminaIR/blob/main/assets/IR_super_resolution.png)

### summary
- current focus it to make a solid Super-resolver before movig to coloration.
- upgrade the flaws and refine the architecture.
