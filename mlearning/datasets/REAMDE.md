## Datasets

### COCO Dataset
#### annotation format

-----------------------------------------------------------------------

### PASCAL Dataset
#### annotation format


-----------------------------------------------------------------------
### DOTA Dataset
#### annotation format

```txt
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
```

For example, 
```txt
2753 2408 2861 2385 2888 2468 2805 2502 plane 0
2870 4250 2916 4268 2912 4283 2866 4263 large-vehicle 0
636 1713 633 1706 646 1698 650 1706 small-vehicle 0
```

##### Directory Structure

<U>The filename of image and txt(label) must be same.</U>
```
- split_dataset
    ├─ train
    │   ├─ images
    |   |   ├─ 1.png
    |   |   ├─ 2.png
    |   |   ├─ ...
    |   |   
    │   └─ labelTxt
    |       ├─ 1.txt
    |       ├─ 2.txt
    |       ├─ ...
    └─ val
        ├─ image
        └─ labelTxt
```


### YOLO Dataset
#### annotation format

```txt
category x y w h 
```

##### Directory Structure

<U>The filename of image and txt(label) must be same.</U>
```
- split_dataset
    ├─ images
    │   ├─ train
    |   |   ├─ 1.png
    |   |   ├─ 2.png
    |   |   ├─ ...
    |   |   
    │   └─ val
    └─ labels
        ├─ train
        |   ├─ 1.txt
        |   ├─ 2.txt
        |   ├─ ...
        └─ val
```




### References:
- [DOTA](https://captain-whu.github.io/DOTA/dataset.html)