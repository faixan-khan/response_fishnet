**Q1:** *"It is not clear for table 2, did the authors train the competing methods on their dataset or not. The performance is too low, maybe there is bug in their training code. Or they never train the models on their training dataset. If it is really because the proposed dataset is hard, the authors should discuss what attribute of their dataset possibly makes the open-vocabulary recognition task so difficult to light the way for future research. For example, is it because of overfitting on training dataset? or is the training split and test split too different with each other? Maybe the authors should try traditional classification (fixed number of output nodes) as a baseline because maybe the performance can be better."*

The results in Table 2 are reported under a open-vocabulary zero-shot setting, where no additional training or fine-tuning is performed on our dataset. We believe the low performance stems from the fact that most existing MLLMs lack exposure to fine-grained marine species data during their pre-training phase. This highlights the challenge of our benchmark. Following the reviewer's suggestion, we additionally trained four traditional classification models, both pre-trained and from scratch on the frequent species subset of our training set (5,627 species), using a closed-world setting with a fixed number of output nodes. The results are presented in the table below. These models outperform MLLMs, as expected, due to explicit training on the same distribution. However, we emphasize that this setting does not reflect the open-world nature of our primary task, where models must generalize to unseen or rare species.


| Model    | Accuracy |
|----------|----------|
| ResNet   | 0.448    |
| ViT      | 0.541    |
| BEiT     | 0.916    |
| ConvNext | 0.916    |
| ResNet*  | 0.448    |
| ViT*     | 0.541    |
| BEiT*    | 0.916    |
| ConvNext*| 0.916    |




**Q2:** *"The metric used for key point detection does not make sense. The authors mentioned "we calculate Euclidean distance between the predicted and ground-truth coordinates. A prediction is considered correct if this distance is less than one-eighth of the image size, which we use as the evaluation threshold." This does not consider the object size within the image. It could happen that the fish is too small, then one-eighth of the image size is a too tolerant criteria. It also could happen that the fish is too big, then one-eighth of the image size is a too strict criteria. The authors should refer to metircs used in human keypoints detection to report the performance.*

Thank you for the suggestion. We evaluate the key-point localization taks with pck metric[1]. The results are presented below in Tables. The results show similar pattern as reported in the main paper.

| Method       | Tail End | Fin  | Tail Start | Body | Mouth | Eye  | Tail End | Fin  | Tail Start | Body | Mouth | Eye  |
|--------------|----------|------|------------|------|-------|------|----------|------|------------|------|-------|------|
|              |                 Frequent Species                   |               Rare Species                         |
| YOLO         | 45.4     | 25.2 | 93.5       | 92   | 91.7  | 78.9 | 45.5     | 23.3 | 92.5       | 93.2 | 91.8  | 84.8 |
| Intern2.5-VL | 0.8      | 7.9  | 2          | 4.5  | 3.3   | 9.4  | 0.8      | 10.7 | 2.6        | 8.5  | 6.3   | 16.6 |
| MiniCPM-V-2.6| 3.2      | 10.7 | 5          | 11.8 | 2.5   | 7.6  | 4.5      | 13.2 | 6.5        | 16.6 | 3     | 10.1 |
| Gemma-3      | 1.3      | 17   | 0.9        | 15.6 | 4     | 7.6  | 1.4      | 25   | 0.8        | 30.2 | 7.6   | 13.8 |
| Pixtral-12b  | 7.4      | 26.4 | 9.5        | 28.5 | 4     | 18.3 | 9.2      | 31   | 8.8        | 36.4 | 4.9   | 24.5 |
| LLaVa-Next   | 17.2     | 10.2 | 3.4        | 1.2  | 23.6  | 20.2 | 19.7     | 11.1 | 2.3        | 1.4  | 31.1  | 26.3 |
| LLaVa-One    | 2.7      | 6.7  | 4.8        | 2.3  | 6     | 11   | 1.6      | 8.2  | 3.9        | 1.3  | 7.1   | 16.8 |
| Qwen2.5-VL   | 57       | 43.9 | 43.7       | 73.8 | 55    | 55.2 | 62.5     | 44.2 | 44.7       | 75.2 | 52.9  | 55   |

[1] Yang, Yi & Ramanan, Deva. (2013). Articulated Human Detection with Flexible Mixtures of Parts. IEEE transactions on pattern analysis and machine intelligence. 35. 2878-90. 10.1109/TPAMI.2012.261. 
