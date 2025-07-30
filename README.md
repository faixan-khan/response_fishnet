### R1 response

**Q1:** *"It is not clear for table 2, did the authors train the competing methods on their dataset or not. The performance is too low, maybe there is bug in their training code. Or they never train the models on their training dataset. If it is really because the proposed dataset is hard, the authors should discuss what attribute of their dataset possibly makes the open-vocabulary recognition task so difficult to light the way for future research. For example, is it because of overfitting on training dataset? or is the training split and test split too different with each other? Maybe the authors should try traditional classification (fixed number of output nodes) as a baseline because maybe the performance can be better."*

The results in Table 2 are reported under a open-vocabulary zero-shot setting, where no additional training or fine-tuning is performed on our dataset. We believe the low performance stems from the fact that most existing MLLMs lack exposure to fine-grained marine species data during their pre-training phase. This highlights the challenge of our benchmark. Following the reviewer's suggestion, we additionally trained three traditional classification models, both pre-trained(* denotes pre-trained on ImageNet) and from scratch on the frequent species subset of our training set (5,627 species), using a closed-world setting with a fixed number of output nodes. The results are presented in the table below. These models outperform MLLMs, as expected, due to explicit training on the same distribution. However, we emphasize that this setting does not reflect the open-world nature of our primary task, where models must generalize to unseen or rare species.


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

### R2 response

**Q1** *"From the provided video, it looks like the detection of keypoints (e.g., body center, fin tips) relies on manual annotation. This is time-intensive, and annotating by several annotators (which I assume was the case) often suffers from inter-annotator variability. It would be worth discussing the potential for automating this step using a keypoint detection framework, e.g., DeepLabCut. Given the existence of the annotated data, training such a model could substantially scale up annotation speed and consistency for future use."*

Yes, we agree that manual annotation of keypoints is a time-intensive process and can be affected by inter-annotator variability. We follow the established pratice of data collection established in [1], which has been widely used for bird classification. And one of the motivations behind releasing this dataset is to enable future efforts toward semi-automated or fully automated annotation. In fact, similar to how we employed GroundedSAM in a semi-automated pipeline to generate segmentation masks for our dataset, the keypoint annotations we provide could serve as valuable training data for frameworks such as DeepLabCut in context of marine species. This can open the door for scaling up annotation efforts more efficiently and consistently in future work.

[1] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. http://www.vision.caltech.edu/visipedia/CUB-200-2011.html, 2011. Dataset and technical report.

**Q2** *"I assume that the 80-20 train-test split strategy may have been chosen due to limited images per species. However, the absence of a validation set raises concerns about potential test set leakage. Without a held-out validation set, different models (or research groups) might report test performance based on differently tuned models, which limits comparability. For a dataset intended to serve as a benchmark, a train-val-test split would be preferable to ensure fair comparisons."*

Thank you for the feedback. The dataset was split with consideration for the number of available images per species. For rare classes (with only one or two images), we included all examples in the test set to ensure they are evaluated, which made it impractical to reserve a validation subset for them. For frequent classes, we used an 80-20 train-test split. Since all MLLMs were evaluated in a strict zero-shot setting, and the trained baseline methods were assessed using k-fold cross-validation. That said, we acknowledge the reviewer’s valid concern regarding comparability across different works. To address this, we will revise the dataset split by introducing a dedicated validation set from the training data, resulting in a 70-10-20 train-val-test split. This will help standardize model tuning and improve fairness and reproducibility for future research using our benchmark.

**Q3** *"I’m also surprised to see only 46.2%-53.6% accuracy at the Family level, even for frequent species. While I haven’t worked specifically on fish taxonomic classification, a quick literature search suggests much higher accuracy. Can the authors comment on what might explain these results?"*

The relatively lower accuracy at the Family level can largely be attributed to the significantly broader taxonomic coverage in our dataset. As shown in Table 1 of the main paper, most prior works focus on a limited set of species and families, which naturally leads to higher accuracy. In contrast, our benchmark includes over 17,000 species spanning 624 families, making the classification task considerably more challenging. Moreover, our evaluation is conducted in a zero-shot setting which further raises the difficulty. Despite this, achieving 46.2%–53.6% accuracy at the Family level is substantially better than random chance (which would be below 0.2%), representing over 300× improvement over random guessing. We believe these results are reasonable given the scale and open-world nature of the task.

**Q4** *"I’m curious about the details of generating the missing morphological descriptions for 13,854 species. Could the authors provide the evaluation details? The 50 descriptions evaluated by experts constitute only 3.6% of the total (missing) amount. Is this sufficient to be certain about the performance? Did all 50 pass the expert check, or were there cases that were unsatisfactory or inaccurate? This would be helpful to include in the supplementary."*

All 50 passed the check, but it should be noted that these descriptions are mimicing those seen in FishBase, they are not full taxonomic descriptions. The threshold for accuracy is therefore fairly low and so high performance is not unexpected.

**Q5** *"It would be especially helpful to see the dataset distribution details, i.e., as a plot for species and their representations, to understand the full picture as well as how many were excluded as underrepresented."*

We ensure that we will include a plot to to show the representation of the species present in our dataset. Since, Neurips does not allow for visual outputs in rebuttal we can not include this for the rebuttal.

**Q6** *"Line 175: Were the 36 images that did not correspond to any known species from the entire taxonomy the ones of species that had very few images? Please provide more details in the supplementary."*

The 36 images did not correspond to any known species, neither from rare or low-frequency categories. It is likely that these represent entirely new, previously undocumented species. Marine experts collaborating on this project are currently investigating these cases to verify their taxonomic status.

**Q7** *"Why is there a switch between YOLO-11 and YOLO-12 as the baseline?"*

We tested both version and they perform similar. We will include Yolo-12 for all the results in the updated paper for consitency.

**Q8** *"Table 1: "FishNet++ provides textual descriptions for more than 35,000 species". I believe this should be "FishNet++ provides more than 35,000 textual descriptions for 17,393 species"."*

We have one description for each species. The descriptions are provided for over 35,000 species that represents all fish species listed on FishBase at the time the work was conducted.

**Q9** *"In tables, separate GPT-4o from Qwen 2.5VL + E-RAG and mark * with an explanation of why you are focusing on the latter, since GPT-4o has higher performance."*

Thank you for the suggestion. We will include a make * for GPT-4o. We focus on Qwen because it is open-source and as GPT is available only via api and not accessible to everyone, we believe we should focus on models that can be used by majority of community to reproduce the results.

**Q10** *"In the context of taxonomic classification, the term 'species identification' is often inaccurate when the task involves genus- or family-level classification. A more appropriate term in these cases would be 'specimen identification'."*

Thank you for the suggestion. We will make the change from species classification to specimen identification in our updated version of the paper.

**Q11** *"The README file does not provide sufficient details. Please make sure that the dataset card is properly filled out."*

Thank you for the feedback. As any changes in dataset are prohibited after submiision, below is an overview of the dataset contents. 

* **`bounding_boxes/`**: Contains one file per image, each storing the bounding box coordinates in the format (left, bottom, right, top) for the corresponding fish instance.

* **`kp_annotations/`**: Includes JSON files where each entry corresponds to an image and contains the (x, y) coordinates for six annotated keypoints.

* **`masks/`**: Provides segmentation masks for each image, highlighting the fish instance at the pixel level.

* **`specie_level/`**: Organized into folders where each folder represents a distinct species class and contains all associated image samples.

* **`description.json:`** A single JSON file with textual descriptions for all species.

### R3 response

**Q1** *"The formulation of fish keypoint localization lacks a rigorous and scientific formulation. The proposed dataset to annotate 6 part annotations (1) Eye location, 2) Mouth location, 3) Pectoral, pelvice and anal fin location, 4) Center of the main body, 5) Tail (caudal fin) start, and 6) Tail end) could be extended to all the fish species? Does such keypoint annotation really satisfy the domain requirements or come from a scientific and true scientific formulation? The authors did not propose a detailed and scientific formulation of such a fish keypoint detection task."*

We begin with these 6 key points because they serve several purposes. Firstly, from a purely logitical standpoint, the six points are relatively easily identifiable across the wide range of images and image qualitites contained in FishNet. Other potential key points such as operculum openings, nares position, fin insertion points, etc. are much harder to label across the range of images in FishNet. Secondly, the 6 key points represent key areas associated with low-resolution morphological differences among species, e.g. relative palcement or eyes, fins, etc. are fundamental measurements in taxonomic descriptions, as are ""missing"" fins with species having different numbers of pelvic fins, or anal fins being absent is some species. Thirdly, the details of these features (e.g. eye size, fin colouration etc.) are often important in species identification, and so providing their locations for development of species identification was deemed potentially useful. We will highlight these rationale more clearly in the revised version of the paper."

**Q2** *"For the Open-Vocabulary Classification task, how can the authors perform the training or evaluation? From the current manuscript, it seems that the authors did not perform the training or fine-tuning of existing VLMs on the proposed dataset. The results in Table 2 and Table 3 are so poor. Why did the authors not perform the fine-tuning on the training set of the proposed dataset? If the authors only conducted the testing on the constructed dataset, the contribution of this work is mainly about dataset construction, lacking the specific novelty regarding the fish monitoring."*

Thank you for the question. As noted, the rare species set contains only 1-2 images per class, and all of them are placed in the test set. This makes it infeasible to train or fine-tune models on the rare set, which was intentionally designed to simulate a real-world open-vocabulary setting where most species are rarely observed or newly encountered. For the frequent classes, we deliberately chose not to fine-tune the VLMs. Our primary motivation was to assess the zero-shot generalization capabilities of current frontier Multimodal Large Language Models (MLLMs) in the marine biology domain. Fine-tuning on the dataset could improve accuracy but often leads to forgetting other abilities, such as object detection or keypoint localization which are also tested on our dataset. We believe a better long-term solution is to incorporate domain-relevant data (like ours) during the pretraining phase, which our dataset now enables. That said, we do include results from several supervised classification baselines below that are trained on the training set and evaluated on the frequent test set(* denotes pre-trained on ImageNet). These show strong performance, but such experiments reflect a closed-world setting, unlike the more challenging open-world generalization that we target for MLLMs. Additionally, our contribution extends beyond just dataset construction. We present three other tasks: segmentation, keypoint localizationan, and detection.We introduce a semi-automated pipeline (using GroundedSAM and GPT-4) to generate high-quality masks and species-level descriptions for over 35,000 marine species. This effort enables scalable, diverse benchmarking and progress in marine monitoring and recognition.

| Model    | Accuracy |
|----------|----------|
| ViT      | 9.9     |
| BEiT     | 10.9    |
| ConvNext | 13.7    |
| ViT*     | 50.3    |
| BEiT*    | 44.8    |
| ConvNext*| 62.1    |

**Q3** *"The constructed detection/segmentation task in this work is too easy. First, the images are center cropped and most of the images only contain one fish. The proposed dataset and benchmark are very far from the real-world fish monitoring and surveying. Furthermore, since each image only contains one instance, it is hard to say the proposed dataset and benchmark could really evaluate the ability of models to localize/ground the required fish based on the given prompts. This problem will be degraded to the classification/matching problem. Meanwhile, the optimized detection and segmentation models will tend to output only one prediction. In the wild condition, the captured image usually contains multiple instances from various species. The reviewer would say the proposed dataset has really limited values. The results in Table 6 demonstrate that the results of existing foundation models could already achieve saturated results."*

The images in the dataset are not center cropped. As shown in Table 6, existing models indeed perform well on detection and segmentation, suggesting they can localize marine species effectively. This makes our dataset a strong starting benchmark to evaluate such capabilities in underwater imagery. However, we emphasize that recognizing the correct species, even with accurate localization, remains a significant challenge, especially given the fine-grained visual distinctions between species. Moreover, tasks like key-point detection, which also don't require explicit domain knowledge, still prove difficult for current models. This highlights the fact that while detection/segmentation may seem saturated, species identification and keypoint localization are far from solved and represent valuable directions for future research. Our dataset is designed to support progress in these areas by providing structured annotations that go beyond coarse detection, to move toward more semantically rich and biologically meaningful benchmarks.

**Q4** *"The reviewer guess the optimized models on the proposed dataset would still yield many false positives and false negatives on the unseen underwater images captured in the wild, with multiple fish species within one image. The authors should perform corresponding experiments."*

The primary contribution of our work is to introduce a richly annotated benchmark tailored to marine species, enabling rigorous evaluation of existing MLLMs and vision models on tasks such as detection, segmentation, keypoint localization, and species recognition. Our results highlight where current models succeed and, more importantly, where they fall short in marine-specific contexts. We agree that testing models on scenes with multiple fish species would be valuable. Acquiring and annotating such data is extremely challenging and resource-intensive, however, our dataset has many instances where multiple fishes are present(~5000) in a single image and various cases where the surrounding underwtaer conditions make it very complex. Nevertheless, we view this as an important future direction and hope that the release of our dataset will encourage broader community efforts toward building and evaluating on such challenging, in-the-wild benchmarks.

**Q5** *"How can the authors make sure the descriptions (descriptive summary) from GPT-4o are reliable or accurate?"*

While it is not possible to guarantee that the GPT-4o-generated summaries are entirely free from hallucinations, we took steps to assess their accuracy. Specifically, we asked marine experts on our team to evaluate the descriptions of 50 marine species they are most familiar with. The experts found the summaries to be effective in capturing the key visual attributes relevant for species recognition in low-resolution scenarios, indicating that the descriptions are generally reliable for our task.


**Q6** *"In Tables 4 and 5, why are the results of these four algorithms (InternVL-2.5, MiniCPM-V-2.6, Gemma-3 and Pixtral-12b) so poor? The authors should provide more explanations regarding these poor results."*

We agree with the reviewer's observation that InternVL-2.5, MiniCPM-V-2.6, Gemma-3, and Pixtral-12b exhibit comparatively poor performance on this fine-grained keypoint detection task for fish. We believe as general-purpose MLLMs are predominantly pre-trained for broader visual understanding, high-level reasoning, and text-to-image alignment. Their pre-training datasets typically lack explicit, dense annotations for precise anatomical keypoints, which require sub-object localization capabilities that differ from standard bounding box detection or image captioning. MiniCPM only includes RefCOCO dataset for grounding, which is not a fine-grained keypoint detection dataset. Both Pixtral-12b and Gemma3 do not release the dataset details but based on their performance it is likely they also do not focus on fine-grained key point detection as a task in their model training. Intern-VL uses standard grounding datasets like RefCOCO and COCO-ReM for grounding but does not deal with point-localization tasks. On the other Qwen2.5-VL specifically focusses on improving accuracy in detecting and pointing. They do so by developing a comprehensive dataset which consists of bounding boxes and point locations with referring expressions. They synthesize the data into various formats, including XML, JSON, and custom formats. We believe, this to be a string factor in Qwen's string performance compared to other MLLMS. Additionally, Qwen2.5-VL uses coordinate values based on the actual dimensions of the input images during training to represent bounding boxes and points. They claim this approach improves the model's ability to capture the real-world scale and spatial relationships of objects. This disparity in performance highlights the importance of our work in demonstrating that even powerful MLLMs require specific fine-tuning (or pre-training) to excel at niche, pixel-precise tasks like keypoint detection, rather than generalizing seamlessly from high-level visual understanding. Our findings highlight Qwen2.5-VL as a particularly promising foundation model for such fine-grained localization challenges in specialized domains. We will include a detailed discussion on the results in our revised paper.

**Q6** *"Are there copyright issues with the images collected from FishBase, iNaturalist, WoRMS and NOAA sources?"*

The newly added images are owned by our collaboraters in this project. There are no issues related to privacy, copyright, and consent.

