**Q1:** *"It is not clear for table 2, did the authors train the competing methods on their dataset or not. The performance is too low, maybe there is bug in their training code. Or they never train the models on their training dataset. If it is really because the proposed dataset is hard, the authors should discuss what attribute of their dataset possibly makes the open-vocabulary recognition task so difficult to light the way for future research. For example, is it because of overfitting on training dataset? or is the training split and test split too different with each other? Maybe the authors should try traditional classification (fixed number of output nodes) as a baseline because maybe the performance can be better."*

The results in Table 2 are reported under a open-vocabulary zero-shot setting, where no additional training or fine-tuning is performed on our dataset. We believe the low performance stems from the fact that most existing MLLMs lack exposure to fine-grained marine species data during their pre-training phase. This highlights the challenge of our benchmark. Following the reviewer's suggestion, we additionally trained three traditional classification models, both pre-trained(deonted by *) and from scratch on the frequent species subset of our training set (5,627 species), using a closed-world setting with a fixed number of output nodes. The results are presented in the table below. These models outperform MLLMs, as expected, due to explicit training on the same distribution. However, we emphasize that this setting does not reflect the open-world nature of our primary task, where models must generalize to unseen or rare species.


| Model    | Accuracy |
|----------|----------|
| ViT      | 9.9     |
| BEiT     | 10.9    |
| ConvNext | 13.7    |
| ViT*     | 50.3    |
| BEiT*    | 44.8    |
| ConvNext*| 62.1    |




**Q2:** *"The metric used for key point detection does not make sense. The authors mentioned "we calculate Euclidean distance between the predicted and ground-truth coordinates. A prediction is considered correct if this distance is less than one-eighth of the image size, which we use as the evaluation threshold." This does not consider the object size within the image. It could happen that the fish is too small, then one-eighth of the image size is a too tolerant criteria. It also could happen that the fish is too big, then one-eighth of the image size is a too strict criteria. The authors should refer to metircs used in human keypoints detection to report the performance.*

Thank you for the suggestion. We evaluate the key-point localization taks with pck metric[1]. The results corresponding to Table 4 in the main paper are presented below with pck(alpha=0.1). The results show similar pattern as in the main paper.

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

**Q3** *"The box detection task and segmentation task in the proposed dataset is almost saturated with existing methods which should not deserve too much discussion or experiments. More disscussion and experiment should be left for the open-vocabulary recognition task which got very low performance."*

While these tasks may appear saturated, we believe they remain important, particularly in the context of explainable fine-grained recognition. Although our dataset does not include part-aware bounding boxes, it provides whole-body bounding boxes and precise keypoint annotations for key parts such as the eyes, mouth, fins, body center, and tail. These part keypoints serve as valuable supervisory signals, and when combined with the whole-body bounding box, they can be used to estimate part regions or sizes, an approach commonly used in part-based recognition literature[1,2]. Such estimates allow researchers to approximate part bounding boxes and enable weakly-supervised part-aware methods. The importance of part-level information in explainable recognition has been well established in prior work, such as [1,2, 3], where aligning part-localized features with textual descriptions improves zero-shot performance. By providing structured part keypoints, our dataset can allow similar explorations in the marine domain, where fine-grained, part-aware distinctions are especially crucial for accurate species identification. While we agree that open-vocabulary classification deserves deeper investigation, we include detection and segmentation tasks to enable holistic and explainable modeling pipelines that future work can build upon.

[1] Link the head to the “beak”: Zero Shot Learning from Noisy Text Description at Part Precision. Mohamed Elhoseiny, Yizhe Zhu, Han Zhang, and Ahmed Elgammal. 
[2] SPDA-CNN: Unifying Semantic Part Detection and Abstraction for Fine-grained Recognition. Han Zhang, Tao Xu, Mohamed Elhoseiny, Xiaolei Huang, Shaoting Zhang, Ahmed Elgammal, and Dimitris Metaxas. 
[3] Multi-Cue Zero-Shot Learning with Strong Supervision. Zeynep Akata, Mateusz Malinowski, Mario Fritz and Bernt Schiele


**Q4** *"Missing some related work of applying computer vision to fishery science: [1] Sea you later: Metadata-guided long-term re-identification for uav-based multi-object tracking [2] Video-based hierarchical species classification for longline fishing monitoring [3] HCIL: Hierarchical Class Incremental Learning for Longline Fishing Visual Monitoring [4] ESA: Expert-and-Samples-Aware Incremental Learning Under Longtail Distribution"*

We thank the reviewer for highlighting these relevant works. [1] addresses multi-object maritime tracking, which, while broader, relates to marine species contexts. [2,3] focus on fish species recognition from longline fishing catches, aligning with our classification goals. [4] explores incremental learning under long-tailed distributions, which is highly applicable to our imbalanced dataset. We will include a discussion of these in the revised paper.

