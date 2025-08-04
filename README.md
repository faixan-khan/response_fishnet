### R1 response

**Q1:** *It’s unclear whether the models in Table 2 were trained on the proposed dataset. The performance seems too low—was there a training bug, or were the models never trained? If the dataset is genuinely difficult, the authors should discuss which attributes contribute to this. For instance, is it due to overfitting, or significant differences between train and test splits? The authors should also consider adding traditional classification (with fixed output nodes) as a baseline, which may perform better.*

The results in Table 2 are reported under an open-vocabulary zero-shot setting, where no additional training or fine-tuning is performed on our dataset. We believe the low performance stems from the fact that most existing MLLMs lack exposure to fine-grained marine species data during their pre-training phase. This highlights the challenge of our benchmark. Following the reviewer's suggestion, we additionally trained three traditional classification models, both pre-trained(* denotes pre-trained on ImageNet) and from scratch on the frequent species subset of our training set (5,627 species), using a closed-world setting with a fixed number of output nodes. The results are presented in the table below. These models outperform MLLMs, as expected, due to explicit training on the same distribution. However, we emphasize that this setting does not reflect the open-world nature of our primary task, where models must generalize to unseen or rare species.


| Model    | Accuracy |
|----------|----------|
| ViT      | 9.9     |
| BEiT     | 10.9    |
| ConvNext | 13.7    |
| ViT*     | 50.3    |
| BEiT*    | 44.8    |
| ConvNext*| 62.1    |




**Q2:** *The metric used for keypoint detection seems flawed. The authors define a correct prediction as one where the Euclidean distance is less than one-eighth of the image size, but this ignores object scale. If the fish is small, the threshold is too lenient; if large, it's too strict. Metrics from human keypoint detection should be considered instead.*

Thank you for the suggestion. We evaluate the key-point localization task with the PCK metric[1]. The results corresponding to Table 4 in the main paper are presented below with pck(alpha=0.1). The results show a similar pattern to that in the main paper.
| Method            |  |              |   Frequent       |         |       |  |      |     |  Rare      |         |    | |
|------------------|------------|------|------------|------|--------|------|------------|------|------------|------|--------|------|
|                         | Tail End    | Fin  | Tail Start  | Body | Mouth  | Eye  | Tail End   | Fin  | Tail Start | Body | Mouth  | Eye  |
| **YOLO**         | 45.4       | 25.2 | 93.5       | 92.0 | 91.7   | 78.9 | 45.5       | 23.3 | 92.5       | 93.2 | 91.8   | 84.8 |
| **Intern2.5-VL** | 0.8        | 7.9  | 2.0        | 4.5  | 3.3    | 9.4  | 0.8        | 10.7 | 2.6        | 8.5  | 6.3    | 16.6 |
| **MiniCPM-V-2.6**| 3.2        | 10.7 | 5.0        | 11.8 | 2.5    | 7.6  | 4.5        | 13.2 | 6.5        | 16.6 | 3.0    | 10.1 |
| **Gemma-3**      | 1.3        | 17.0 | 0.9        | 15.6 | 4.0    | 7.6  | 1.4        | 25.0 | 0.8        | 30.2 | 7.6    | 13.8 |
| **Pixtral-12b**  | 7.4        | 26.4 | 9.5        | 28.5 | 4.0    | 18.3 | 9.2        | 31.0 | 8.8        | 36.4 | 4.9    | 24.5 |
| **LLaVa-Next**   | 17.2       | 10.2 | 3.4        | 21.0 | 23.6   | 20.2 | 19.7       | 11.1 | 2.3        | 9.0  | 31.1   | 26.3 |
| **LLaVa-One**    | 2.7        | 6.7  | 4.8        | 2.3  | 6.0    | 11.0 | 1.6        | 8.2  | 3.9        | 1.3  | 7.1    | 16.8 |
| **Qwen2.5-VL**   | 57.0       | 43.9 | 43.7       | 73.8 | 55.0   | 55.2 | 62.5       | 44.2 | 44.7       | 75.2 | 52.9   | 55.0 |




[1] Yang, Yi & Ramanan, Deva. (2013). Articulated Human Detection with Flexible Mixtures of Parts. IEEE transactions on pattern analysis and machine intelligence. 35. 2878-90. 10.1109/TPAMI.2012.261. 

**Q3** *The box detection task and segmentation task in the proposed dataset is almost saturated with existing methods which should not deserve too much discussion or experiments. More disscussion and experiment should be left for the open-vocabulary recognition task which got very low performance.*

While these tasks may appear saturated, we believe they remain important, particularly in the context of explainable fine-grained recognition. Although our dataset does not include part-aware bounding boxes, it provides whole-body bounding boxes and precise keypoint annotations for key parts such as the eyes, mouth, fins, body center, and tail. These part keypoints serve as valuable supervisory signals, and when combined with the whole-body bounding box, they can be used to estimate part regions or sizes, an approach commonly used in part-based recognition literature[1,2]. Such estimates allow researchers to approximate part bounding boxes and enable weakly-supervised part-aware methods. The importance of part-level information in explainable recognition has been well established in prior work, such as [1,2, 3], where aligning part-localized features with textual descriptions improves zero-shot performance. By providing structured part keypoints, our dataset can allow similar explorations in the marine domain, where fine-grained, part-aware distinctions are especially crucial for accurate species identification. While we agree that open-vocabulary classification deserves deeper investigation, we include detection and segmentation tasks to enable holistic and explainable modeling pipelines that future work can build upon.

[1] Link the head to the “beak”: Zero Shot Learning from Noisy Text Description at Part Precision. Mohamed Elhoseiny, Yizhe Zhu, Han Zhang, and Ahmed Elgammal. 

[2] SPDA-CNN: Unifying Semantic Part Detection and Abstraction for Fine-grained Recognition. Han Zhang, Tao Xu, Mohamed Elhoseiny, Xiaolei Huang, Shaoting Zhang, Ahmed Elgammal, and Dimitris Metaxas. 

[3] Multi-Cue Zero-Shot Learning with Strong Supervision. Zeynep Akata, Mateusz Malinowski, Mario Fritz and Bernt Schiele


**Q4** *Missing some related work of applying computer vision to fishery science: [1] Sea you later: Metadata-guided long-term re-identification for uav-based multi-object tracking [2] Video-based hierarchical species classification for longline fishing monitoring [3] HCIL: Hierarchical Class Incremental Learning for Longline Fishing Visual Monitoring [4] ESA: Expert-and-Samples-Aware Incremental Learning Under Longtail Distribution*

We thank the reviewer for highlighting these relevant works. [1] addresses multi-object maritime tracking, which, while broader, relates to marine species contexts. [2,3] focus on fish species recognition from longline fishing catches, aligning with our classification goals. [4] explores incremental learning under long-tailed distributions, which is highly applicable to our imbalanced dataset. We will include a discussion of these in the revised paper.


### R2 response

**Q1** *From the provided video, it looks like the detection of keypoints (e.g., body center, fin tips) relies on manual annotation. This is time-intensive, and annotating by several annotators (which I assume was the case) often suffers from inter-annotator variability. It would be worth discussing the potential for automating this step using a keypoint detection framework, e.g., DeepLabCut. Given the existence of the annotated data, training such a model could substantially scale up annotation speed and consistency for future use.*

Yes, we agree that manual annotation of keypoints is a time-intensive process and can be affected by inter-annotator variability. We follow the established practice of data collection established in [1], which has been widely used for bird classification. And one of the motivations behind releasing this dataset is to enable future efforts toward semi-automated or fully automated annotation. In fact, similar to how we employed GroundedSAM in a semi-automated pipeline to generate segmentation masks for our dataset, the keypoint annotations we provide could serve as valuable training data for frameworks such as DeepLabCut in the context of marine species. This can open the door for scaling up annotation efforts more efficiently and consistently in future work.

[1] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. http://www.vision.caltech.edu/visipedia/CUB-200-2011.html, 2011. Dataset and technical report.

**Q2** *I assume that the 80-20 train-test split strategy may have been chosen due to limited images per species. However, the absence of a validation set raises concerns about potential test set leakage. Without a held-out validation set, different models (or research groups) might report test performance based on differently tuned models, which limits comparability. For a dataset intended to serve as a benchmark, a train-val-test split would be preferable to ensure fair comparisons.*

Thank you for the feedback. The dataset was split with consideration for the number of available images per species. For rare classes (with only one or two images), we included all examples in the test set to ensure they are evaluated, which made it impractical to reserve a validation subset for them. For frequent classes, we used an 80-20 train-test split. Since all MLLMs were evaluated in a strict zero-shot setting, and the trained baseline methods were assessed using k-fold cross-validation. That said, we acknowledge the reviewer’s valid concern regarding comparability across different works. To address this, we will revise the dataset split by introducing a dedicated validation set from the training data, resulting in a 70-10-20 train-val-test split. This will help standardize model tuning and improve fairness and reproducibility for future research using our benchmark.

**Q3** *I’m also surprised to see only 46.2%-53.6% accuracy at the Family level, even for frequent species. While I haven’t worked specifically on fish taxonomic classification, a quick literature search suggests much higher accuracy. Can the authors comment on what might explain these results?*

The relatively lower accuracy at the Family level can largely be attributed to the significantly broader taxonomic coverage in our dataset. As shown in Table 1 of the main paper, most prior works focus on a limited set of species and families, which naturally leads to higher accuracy. In contrast, our benchmark includes over 17,000 species spanning 624 families, making the classification task considerably more challenging. Moreover, our evaluation is conducted in a zero-shot setting, which further raises the difficulty. Despite this, achieving 46.2%–53.6% accuracy at the Family level is substantially better than random chance (which would be below 0.2%), representing over 300× improvement over random guessing. We believe these results are reasonable given the scale and open-world nature of the task.

**Q4** *I’m curious about the details of generating the missing morphological descriptions for 13,854 species. Could the authors provide the evaluation details? The 50 descriptions evaluated by experts constitute only 3.6% of the total (missing) amount. Is this sufficient to be certain about the performance? Did all 50 pass the expert check, or were there cases that were unsatisfactory or inaccurate? This would be helpful to include in the supplementary.*

All 50 passed the check. However, it is important to clarify that these summaries are designed to mimic the style and content of FishBase descriptions rather than serve as formal taxonomic entries. As such, the threshold for accuracy is lower, focusing more on capturing key visual cues relevant for species recognition.

**Q5** *It would be especially helpful to see the dataset distribution details, i.e., as a plot for species and their representations, to understand the full picture as well as how many were excluded as underrepresented.*

We ensure that we will include a plot to show the representation of the species present in our dataset. Since Neurips does not allow for visual outputs in rebuttal, we can not include this for the rebuttal.

**Q6** *Line 175: Were the 36 images that did not correspond to any known species from the entire taxonomy the ones of species that had very few images? Please provide more details in the supplementary.*

The 36 images did not correspond to any known species, neither from rare nor from low-frequency categories. It is likely that these represent entirely new, previously undocumented species. Marine experts collaborating on this project are currently investigating these cases to verify their taxonomic status.

**Q7** *Why is there a switch between YOLO-11 and YOLO-12 as the baseline?*

We chose the latest YOLO model for each task. As pre-trained weights have not been made publicly available for YOLO-12 for segmentation, we chose YOLO-11 for the segmentation task. 

**Q8** *Table 1: "FishNet++ provides textual descriptions for more than 35,000 species". I believe this should be "FishNet++ provides more than 35,000 textual descriptions for 17,393 species".*

We have one description for each species. The descriptions are provided for over 35,000 species that represent all fish species listed on FishBase at the time the work was conducted.

**Q9** *In tables, separate GPT-4o from Qwen 2.5VL + E-RAG and mark * with an explanation of why you are focusing on the latter, since GPT-4o has higher performance.*

Thank you for the suggestion. We will include a make * for GPT-4o. We focus on Qwen because it is open-source, and as GPT is available only via api and not accessible to everyone, we believe we should focus on models that can be used by the majority of the community to reproduce the results.

**Q10** *In the context of taxonomic classification, the term 'species identification' is often inaccurate when the task involves genus- or family-level classification. A more appropriate term in these cases would be 'specimen identification'.*

Thank you for the suggestion. We will make the change from species classification to specimen identification in our updated version of the paper.

**Q11** *The README file does not provide sufficient details. Please make sure that the dataset card is properly filled out.*

Thank you for the feedback. As any changes in the dataset are prohibited after submission, below is an overview of the dataset contents. 

* **`bounding_boxes.zip`**: Contains one file per image, each storing the bounding box coordinates in the format (left, bottom, right, top) for the corresponding fish instance.

* **`kp_annotations.zip`**: Includes JSON files where each entry corresponds to an image and contains the (x, y) coordinates for six annotated keypoints.

* **`masks.zip`**: Provides segmentation masks for each image, highlighting the fish instance at the pixel level.

* **`specie_level.zip`**: Organized into folders where each folder represents a distinct species class and contains all associated image samples.

* **`description.json:`** A single JSON file with textual descriptions for all species.

###R3 response

**Q1** *The formulation for fish keypoint localization lacks scientific rigor. It's unclear if the six annotated parts (eye, mouth, fins, body center, tail start, and tail end) can be consistently applied across all species or if they meet domain-specific standards. A more detailed and scientifically grounded justification for these keypoints is needed.*

We begin with these 6 key points because they serve several purposes. Firstly, from a purely logical standpoint, the six points are relatively easily identifiable across the wide range of images and image qualities contained in FishNet. Other potential key points, such as operculum openings, nares position, fin insertion points, etc., are much harder to label across the range of images in FishNet. Secondly, the 6 key points represent key areas associated with low-resolution morphological differences among species, e.g., relative placement of eyes, fins, etc., are fundamental measurements in taxonomic descriptions, as are "missing" fins with species having different numbers of pelvic fins, or anal fins being absent in some species. Thirdly, the details of these features (e.g., eye size, fin colouration, etc.) are often important in species identification, and so providing their locations for the development of species identification was deemed potentially useful. We will highlight this rationale more clearly in the revised version of the paper."

**Q2** *For the Open-Vocabulary Classification task, it's unclear how training or evaluation was conducted. The manuscript suggests no fine-tuning of existing VLMs on the proposed dataset, and results in Tables 2 and 3 are poor. Why wasn’t fine-tuning performed? If only testing was done, the contribution appears limited to dataset construction, lacking novel insights into fish monitoring.*

Thank you for the question. As noted, the rare species set contains only 1-2 images per class, and all of them are placed in the test set. This makes it infeasible to train or fine-tune models on the rare set, which was intentionally designed to simulate a real-world open-vocabulary setting where most species are rarely observed or newly encountered. For the frequent classes, we deliberately chose not to fine-tune the VLMs. Our primary motivation was to assess the zero-shot generalization capabilities of current frontier Multimodal Large Language Models (MLLMs) in the marine biology domain. Fine-tuning on the dataset could improve accuracy, but often leads to forgetting other abilities, such as object detection or keypoint localization, which are also tested on our dataset. We believe a better long-term solution is to incorporate domain-relevant data (like ours) during the pretraining phase, which our dataset now enables. That said, we do include results from several supervised classification baselines below that are trained on the training set and evaluated on the frequent test set(* denotes pre-trained on ImageNet). These show strong performance, but such experiments reflect a closed-world setting, unlike the more challenging open-world generalization that we target for MLLMs. Additionally, our contribution extends beyond just dataset construction. We present three other tasks: segmentation, keypoint localization, and detection. We introduce a semi-automated pipeline (using GroundedSAM and GPT-4) to generate high-quality masks and species-level descriptions for over 35,000 marine species. This effort enables scalable, diverse benchmarking and progress in marine monitoring and recognition.

| Model    | Accuracy |
|----------|----------|
| ViT      | 9.9     |
| BEiT     | 10.9    |
| ConvNext | 13.7    |
| ViT*     | 50.3    |
| BEiT*    | 44.8    |
| ConvNext*| 62.1    |

**Q3** *The constructed detection/segmentation task appears too easy. Images are center-cropped with only one fish. This setup deviates from real-world fish monitoring scenarios, where images often contain multiple species. As a result, the task becomes more like classification/matching rather than true localization. Optimized models may default to predicting a single instance, limiting the benchmark's ability to evaluate grounding capabilities. In the wild, such models would likely struggle with false positives/negatives. Moreover, results in Table 6 suggest that existing models already achieve near-saturated performance.*

The images in the dataset are not center-cropped. As shown in Table 6, existing models indeed perform well on detection and segmentation, suggesting they can localize marine species effectively. This makes our dataset a strong starting benchmark to evaluate such capabilities in underwater imagery. However, we emphasize that recognizing the correct species, even with accurate localization, remains a significant challenge, especially given the fine-grained visual distinctions between species. Moreover, tasks like key-point detection, which also don't require explicit domain knowledge, still prove difficult for current models. This highlights the fact that while detection/segmentation may seem saturated, species identification and keypoint localization are far from solved and represent valuable directions for future research. Our dataset is designed to support progress in these areas by providing structured annotations that go beyond coarse detection, to move toward more semantically rich and biologically meaningful benchmarks.

**Q4** *The reviewer guess the optimized models on the proposed dataset would still yield many false positives and false negatives on the unseen underwater images captured in the wild, with multiple fish species within one image. The authors should perform corresponding experiments.*

The primary contribution of our work is to introduce a richly annotated benchmark tailored to marine species, enabling rigorous evaluation of existing MLLMs and vision models on tasks such as detection, segmentation, keypoint localization, and species recognition. Our results highlight where current models succeed and, more importantly, where they fall short in marine-specific contexts. We agree that testing models on scenes with multiple fish species would be valuable. Acquiring and annotating such data is extremely challenging and resource-intensive; however, our dataset has many instances where multiple fish are present(~5000) in a single image, and various cases where the surrounding underwater conditions make it very complex. Nevertheless, we view this as an important future direction and hope that the release of our dataset will encourage broader community efforts toward building and evaluating on such challenging, in-the-wild benchmarks.

**Q5** *How can the authors make sure the descriptions (descriptive summary) from GPT-4o are reliable or accurate?*

While it is not possible to guarantee that the GPT-4o-generated summaries are entirely free from hallucinations, we took steps to assess their accuracy. Specifically, we asked marine experts on our team to evaluate the descriptions of 50 marine species they are most familiar with. The experts found the summaries to be effective in capturing the key visual attributes relevant for species recognition in low-resolution scenarios, indicating that the descriptions are generally reliable for our task.


**Q6** *In Tables 4 and 5, why are the results of these four algorithms (InternVL-2.5, MiniCPM-V-2.6, Gemma-3 and Pixtral-12b) so poor? The authors should provide more explanations regarding these poor results.*

We agree with the reviewer's observation that InternVL-2.5, MiniCPM-V-2.6, Gemma-3, and Pixtral-12b exhibit comparatively poor performance on this fine-grained keypoint detection task for fish. We believe that general-purpose MLLMs are predominantly pre-trained for broader visual understanding, high-level reasoning, and text-to-image alignment. Their pre-training datasets typically lack explicit, dense annotations for precise anatomical keypoints, which require sub-object localization capabilities that differ from standard bounding box detection or image captioning. MiniCPM only includes the RefCOCO dataset for grounding, which is not a fine-grained keypoint detection dataset. Both Pixtral-12b and Gemma3 do not release the dataset details, but based on their performance, it is likely that they also do not focus on fine-grained key point detection as a task in their model training. Intern-VL uses standard grounding datasets like RefCOCO and COCO-ReM for grounding, but does not deal with point-localization tasks. On the other hand, Qwen2.5-VL specifically focuses on improving accuracy in detecting and pointing. They do so by developing a comprehensive dataset that consists of bounding boxes and point locations with referring expressions. They synthesize the data into various formats, including XML, JSON, and custom formats. We believe this to be a strong factor in Qwen's string performance compared to other MLLMS. Additionally, Qwen2.5-VL uses coordinate values based on the actual dimensions of the input images during training to represent bounding boxes and points. They claim this approach improves the model's ability to capture the real-world scale and spatial relationships of objects. This disparity in performance highlights the importance of our work in demonstrating that even powerful MLLMs require specific fine-tuning (or pre-training) to excel at niche, pixel-precise tasks like keypoint detection, rather than generalizing seamlessly from high-level visual understanding. Our findings highlight Qwen2.5-VL as a particularly promising foundation model for such fine-grained localization challenges in specialized domains. We will include a detailed discussion on the results in our revised paper.

**Q7** *Are there copyright issues with the images collected from FishBase, iNaturalist, WoRMS and NOAA sources?*

The newly added images are owned by our collaborators in this project. There are no issues related to privacy, copyright, or consent.

###R4 response

**Q1** *The trained models in each experiment should be highlighted/denoted more clearly in the results tables e.g. Tables 4,5,6. In particular, table 6 does not have a line or anything to show whether the YOLO has been finetuned. The rows which have been finetuned should have that written clearly.*

Thank you for your suggestion. We will update the paper to include this. For Key-point localization, detection, and segmentation tasks, Yolo was trained on our dataset, and the remaining methods, all of which are large models pre-trained on internet-scale data, were tested in a zero-shot setting.

**Q2** *In Section 3.5, the claim that training a YOLO model on the ground truth masks proves that the semi-automated pipeline is effective is false: you have only proved that YOLO can learn the masks that have been provided, it does not mean that those are masks are actually accurate. The ground truth of this assessment is still an output of the semi-automated pipeline. This claim should be corrected.*

We would like to clarify that for the test set, the masks generated by the semi-automated pipeline are subsequently manually verified and corrected by human annotators. Any mask that does not accurately align with the object is corrected to ensure high-quality ground truth. Therefore, we trained a YOLO model on masks obtained from our semi-automated pipeline and evaluated on verified masks, demonstrating strong performance, which provides evidence that our semi-automated pipeline is effective and produces usable masks.

**Q3** *38,326 out of the total images are taken above the water – this is a limitation because a model trained on these images will not be likely to perform well for fish identification from BRUVs, ROVs or underwater towed camera platforms.*

This is a strength, not a limitation. Many tasks in fish science are associated with ex-situ and in-situ species identification. Examples include fish market monitoring, fish landings monitoring, and remote electronic monitoring of fisheries. Humans are able to identify fish both in-situ and ex-situ, with the knowledge base and skill set required to do so being highly transferable, a data set that allows training and generalisation across both in-situ and ex-situ images is more likely to replicate human capability than one only trained on images from one or the other. We will include this discussion in the revised paper.

**Q4** *Fish can often have different visual characteristics for male/female fish – was this considered in the annotation of the dataset or in the creation of the textual descriptors? These descriptions would likely be different depending on the sex of the fish, therefore introducing noise into the dataset if a description is used alongside an image of the wrong sex for downstream training.*

We agree with the reviewer. This challenge along with the differences between ontogenetic stages (e.g. larval forms vs juvenile forms vs adult forms) is something we would like to build on in future, but the dataset simply isn't rich enough for the vast majority of species to effectively tackle this challenge yet. And collecting a dataset that can show such a difference and annotating it is a very hard and time-consuming task that can only be done by experts in marine biology.

**Q5** *README in huggingface does not provide explanations/instructions for the dataset usage. It is unclear how to access the images – the annotations seem to be present in .zip files, but the images are not obvious.*

Thank you for the feedback. As any changes in the dataset are prohibited after submission, below is an overview of the dataset contents. 

* **`bounding_boxes.zip`**: Contains one file per image, each storing the bounding box coordinates in the format (left, bottom, right, top) for the corresponding fish instance.

* **`kp_annotations.zip`**: Includes JSON files where each entry corresponds to an image and contains the (x, y) coordinates for six annotated keypoints.

* **`masks.zip`**: Provides segmentation masks for each image, highlighting the fish instance at the pixel level.

* **`specie_level.zip`**: Organized into folders where each folder represents a distinct species class and contains all associated image samples.

* **`description.json:`** A single JSON file with textual descriptions for all species.


### R2 New response


**Q1 Part 1**  *The author did not answer my question directly: whether their key point definition has a biologically meaningful and valuable formulation? Can this formulation be applied to all fish species, since some fish species do not have fins or tails?*


We note that the original comment asked whether the key points satisfied the scientific requirements, to which we have responded. They are justified primarily from a taxonomic identification perspective, both in terms of their relative location (including their absence) being important parts of low-resolution taxonomy, and of the details of them being important (e.g., colours, patterning, size) at higher resolutions. With taxonomy being the key challenge and concern of this dataset (and the subfield as a whole).  

From a biologically meaningful point of view, the presence/absence of fins is meaningful in fish locomotion, and thus movement ecology and ecological lifestyle, as is their relative positioning, which contributes to these. The caudal fin morphology, in particular, is commonly used as a predictor of metabolic demand, which is in turn associated with individual and population growth rates. Similarly, position, size, and morphology of eyes are a strong predictor of ecological lifestyle (eg, benthic vs benthopelagic vs pelagic) predation strategy (eg, ambush vs raptorial). We are happy to add this additional detail to the revised draft.

With respect to the lack of fins, tails, or other key points in some species. Their absence is as (if not more) important to taxonomic identification and biological questions as their presence. It must be stressed that fish represent the oldest and most diverse vertebrate lineage on earth. They comprise more than 50% of all known vertebrate species globally. The taxa contains so much morphological diversity that, if we were to label key points common to and present in all species, we would likely be left with the center body mass as the only possible one.



**Q1 Part 2** *Also, I also noticed the reviewer V8SB had a concern regarding the evaluation metric for the keypoint localization. The author responded: "The distance thresholds used are relative to the object's bounding box size and are set at values like 0.1, 0.2, 0.3, 0.4, and 0.5" without any further details. Considering these two factors, I feel the current task formulation and evaluation metric regarding the fish keypoint localization have some non-ignorable issues.*

We acknowledge the reviewer's concern regarding the evaluation metric for fish keypoint localization. To clarify, we use the Percentage of Correct Keypoints (PCK) metric, which is a widely adopted standard in keypoint localization tasks. The PCK metric determines a keypoint prediction to be correct if the Euclidean distance between the predicted keypoint and the ground-truth keypoint is less than a threshold, defined as α × max(object width, object height) or α × diagonal length of the object bounding box.

In our case, we use the normalized diagonal formulation, where a prediction is considered correct if the distance is within α × object diagonal length. Specifically, we report PCK at α = 0.1, which is a strict threshold and commonly considered a strong indicator of localization accuracy. This setting ensures that only highly accurate predictions are counted as correct. We follow the formulation explained in [1], as used in the Waymo Open Dataset, where the diagonal-normalized PCK is employed to fairly compare keypoint predictions across objects of varying sizes.

[1] Waymo Research. (n.d.). Pose Estimation Metric - Waymo Open Dataset. Retrieved from https://github.com/waymo-research/waymo-open-dataset


**Q2** *Regarding your supervised close-set classification, how many species were used for your training and why did the pre-training on the ImageNet dataset lead to such performance improvements (13.7 to 62.1 for ConvNext). Since the ImageNet dataset does not contain too many fish species, it does not make sense that the pre-training on the ImageNet dataset will lead to a big performance improvement.*

We thank the reviewer for raising this important point. While it is true that the ImageNet dataset does not contain many fish species, the benefits of ImageNet pretraining extend far beyond class-specific knowledge. Pretraining on large-scale datasets like ImageNet allows models to learn a rich hierarchy of generic visual features, including:
* **Low-level features** such as edges, textures, and color gradients
* **Mid-level features** like shapes, patterns, and contours
* **High-level compositional cues** useful for object understanding

These features serve as a strong initialization for a wide range of downstream tasks, even in domains that differ from ImageNet's object categories, such as underwater imagery. In fact, even without any fine-tuning, simple linear probing on ImageNet-pretrained features achieves strong performance. For instance, in [1], linear probing on a model pretrained with a joint-embedding architecture achieved 47.6% accuracy on iNaturalist classification, which clearly shows that such features generalize well despite the domain gap.

This effect is further corroborated in the FishNet benchmark [2], where a ConvNeXt model pretrained on ImageNet achieved 87.7% accuracy across 'common' and' medium' family splits, which corresponds to our 'frequent' species set, compared to only 34.6% when trained from scratch. Thus, while ImageNet may not cover fish species explicitly, the transferable visual priors it offers significantly boost performance in fine-grained domains like marine species classification.


[1] Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.

[2] Khan, F. F., Li, X., Temple, A. J., & Elhoseiny, M. FishNet: A Large-scale Dataset and Benchmark for Fish Recognition, Detection, and Functional Trait Prediction.

**Q3 Part 1** *From the provided visual example in Fig. 1, it is indeed that the images are center-cropped, where the fish is localized at the center part.*

We agree with the reviewer. While some images in our dataset appear center-cropped, it is not the case that all images are centered or uniformly framed. We will highlight images with such cases in the revised draft.


**Q3 Part 2** *The over-saturated results in Table 6 reveal that it is not easy to perform detection or segmentation for existing algorithms. The formulated task in this work will degrade to a classification task.*

We believe the reviewer meant to say it is easy for existing algorithms. 

The value of a dataset like FishNet++, with its focus on individual fish images, lies in its foundational "concept-first" approach to learning. This strategy mirrors how both humans and methods like [1] have been shown to help detection by mastering concepts before tackling other problems. Similar to how ImageNet helped improve detection in [1], our dataset can be used to improve detection for marine species. Our collaborators collected the images to study the different fish species; therefore, these images are of rare species of high quality. 

[1] Joseph Redmon and Ali Farhadi. YOLO9000: Better, Faster, Stronger. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6517–6525, 2017. 


**Q4 Part1** *Your whole dataset has 99,556 images, but only around 5,000 images have more than one fish. I would say the ratio (around 1/20) is quite low.* 

We appreciate the reviewer's concern. While it is true that only a subset of our dataset (~5,000 out of 99,556 images) contains multiple fish, this reflects the inherent difficulty in collecting high-quality, naturally occurring underwater imagery of marine species. In practice, automated underwater monitoring systems, which are a primary source for FishNet++ and most marine related datasets, often record vast amounts of footage where no species are present, or the visibility is too poor for usable annotations. As a result, images with multiple clearly visible and identifiable fish are relatively rare, despite large-scale data collection efforts.

Moreover, as our classification results indicate, current models already struggle with single-instance recognition due to fine-grained inter-species similarity and significant visual variability. Extending this to multi-instance scenes is a much harder problem, not only due to potential occlusion or clutter, but also because multiple fish in a single image often belong to different species, further increasing the task complexity.

While our current dataset may have a lower proportion of multi-fish images, we believe it serves as a valuable and realistic foundation. It enables robust single-instance learning and provides a stepping stone for future work in multi-instance object detection within the challenging marine domain.

**Q4 Part2** *The authors also did not provide any supportive evidence to address my concern regarding the false negatives and false positives. I am not convinced by the response from the authors.*

In traditional object detection tasks, the model must answer both the "where" (localization) and "what" (classification) questions. However, in our evaluation, we primarily focused on the "where" component, i.e., whether the model can accurately localize fish instances without enforcing species-level classification. This was done due to the poor performance in the species classification task.

So, for cases with multiple instances of multiple species in an image, it is likely that errors in species recognition would dominate. However, this does not necessarily indicate a failure in localization performance.

As shown in our experiments, QWEN-2.5-VL shows strong performance in the detection task. Furthermore, Qwen-2.5-VL also performs well in our keypoint localization task, which reinforces its strong visual perception and spatial reasoning abilities. Based on these results, we have no strong reason to believe that models like QWEN-2.5-VL would fail in multi-instance localization scenarios, especially when species classification is decoupled from detection. While our dataset has relatively fewer multi-instance images, the results suggest that existing models possess the capabilities needed to localize multiple fish.

Given that the majority of our dataset consists of single-instance images, it is reasonable to expect that the YOLO baseline model trained on it may face limitations when applied to multi-instance scenarios. However, such models can still be highly valuable in several practical applications. For example: 1)Automated Fish Sorting and Grading Systems: In industrial aquaculture settings, fish often pass individually along conveyor belts or through pipes and channels, where single-instance detection models can be used to sort or grade fish based on size. 2)Individual Fish Monitoring in Research and Aquariums: In controlled environments such as research tanks, breeding programs, or aquariums, fish are often isolated or easily distinguishable. Here, single-instance models can support tasks such as behavior tracking, growth monitoring, and health assessment.


**Q5** *I suggest that the authors add some statistics to provide a rough estimation regarding the caption accuracy. For example, the domain experts evaluate 500 or 1,000 captions from GPT-4o regarding different aspects and report the accuracy.*

The descriptions were not directly generated by GPT-4o. Instead, species-specific information was first collected from reliable sources such as FishBase and other authoritative references. GPT-4o was then used to extract visually discriminative attributes from these expert-verified texts to aid in classification. This approach ensures that the content is grounded in domain expertise, with GPT-4o's role focused on structuring the information to highlight visual cues relevant for downstream tasks. 

Moreover, below we show statistically that with a 95% confidence interval, 94.4%  of generated descriptions are statistically correct.



We have a population of ~35,000 items, from which we randomly sampled 50 descriptions, and all were deemed correct by marine experts. 
We want to estimate the **true accuracy** of the entire dataset with statistical confidence.

- Population size: **35,000**
- Sample size: **n = 50**
- Observed accuracy in sample: **100%**
- Let $\hat{p} = 1$ be the sample accuracy
- Let $P$ be the true (unknown) population accuracy
- We aim to compute a **one-sided 95% lower confidence bound** on $P$
- Critical value for 95% confidence (one-sided): $Z = 1.65$

We use the normal approximation to the binomial:

$$\frac{\hat{p} - P}{\sqrt{\frac{P(1 - P)}{n}}} \sim \mathcal{N}(0, 1)$$

Substitute $\hat{p} = 1$ and $n = 50$:

$$\frac{1 - P}{\sqrt{\frac{P(1 - P)}{50}}} < 1.65$$

Multiply both sides by $\sqrt{\frac{P(1 - P)}{50}}$:

$$1 - P < 1.65 \cdot \sqrt{\frac{P(1 - P)}{50}}$$

Divide both sides by $\sqrt{1-P}$:

$$\sqrt{1 - P} < \frac{1.65}{\sqrt{50}} \cdot \sqrt{P}$$

Square both sides:

$$1 - P < \left( \frac{1.65^2}{50} \right) P$$

$$1 < P \left( 1 + \frac{1.65^2}{50} \right)$$

$$P > \frac{1}{1 + \frac{1.65^2}{50}} \approx 0.944$$

With 95% confidence, **at least 94.4% of our full dataset is correct**.

Q7): Please respond to the Ethics Reviews.

Thank you for the reminder. We have responded to the Ethics Reviews.


### ETHICS


**1** *Checklist Q12 claims “Yes – all the assets used have been properly credited” . No concrete licence table appears in the paper or appendix.*

All images in the dataset are either sourced from FishNet or provided by our project collaborators. FishNet's license permits free use of its images and it has been cited, and for the remaining data, our collaborators are the rightful owners. The dataset will be publicly available under a public license which will according to the Neurips policies.

**2** *Main text: GPT-4o used to generate 35 k species descriptions . Checklist Q16, however, marks “NA” for LLM usage.*

We apologize for the oversight. There was a misunderstanding, we initially interpreted the checklist item as referring solely to LLM usage in paper writing. However, we acknowledge that GPT-4o was used to generate species descriptions, and we will update the checklist accordingly to accurately reflect its use.

**3** *No mention of stripping GPS or other metadata surfaced in the manuscript or checklist.*

We will ensure all the newly added images are stripped of any metadata containing any information that can be used to locate the source of the image.

**4** *Checklist Q10 “Broader impacts” = “NA” ; safeguard question likewise “NA”*

We appreciate the reviewer’s important observation regarding the potential misuse of automated species recognition systems. Our primary motivation behind FishNet++ is to support biodiversity monitoring, conservation, and scientific research. We fully recognize that any powerful technology may have dual-use implications.

* **Data Licensing and Access Control**  All data included in FishNet++ either comes from open sources with appropriate licenses (e.g., FishNet) or from collaborators that support conservation and academic research. Access to sensitive metadata (e.g., geolocation) will be excluded to prevent misuse.
* **Intended Use Guidelines** We will clearly outline intended-use cases and responsible usage guidelines in the dataset documentation, emphasizing that FishNet++ is designed for scientific, conservation, and educational purposes.
* **Collaboration with Conservation Entities** Our dataset is being developed in collaboration with marine biologists who are deeply involved in protecting marine biodiversity. Their expertise helps ensure alignment with ethical standards.


**5** *Authors note long-tail skew and add 5 k images from under-represented regions.*

The long-tail distribution reflects the natural occurrence of marine species, some species, especially those found in deep-sea environments, are inherently rare and difficult to photograph. In contrast, shallow-water species are more abundant and easier to capture. To partially mitigate this imbalance, our collaborators undertook the time-consuming task of collecting 5,000 images of under-represented deep-sea species. However, the long-tail remains a fundamental characteristic of the domain, and our dataset mirrors this natural distribution.

**6** *No CO₂ or energy accounting found.*

Our project utilized GPT-4o for inference to extract discriminative visual descriptions from provided information for approximately 35,000 unique species. The direct, granular CO₂ emissions and energy consumption data for our specific GPT-4o inference requests were not provided by OpenAI. This is a common limitation due to the proprietary nature of large-scale AI infrastructure and the multi-tenant environment in which these models operate.

Despite this, we can estimate our energy consumption based on publicly available research. According to recent analyses, such as those from Epoch AI[1] and [2], a typical GPT-4o query consumes approximately 0.3 watt-hours (Wh). For our 35,000 inferences, this would translate to an estimated energy consumption of 10,500 Wh (10.5 kWh).

Recognizing that query complexity can influence energy usage, if our longer queries were to consume, for example, 0.5 Wh each (a plausible upper estimate for text-based inference), the total energy consumption would rise to approximately 17,500 Wh (17.5 kWh).

To contextualize this, 17.5 kWh is roughly equivalent to the energy contained in about half a gallon of gasoline, which for a typical passenger car translates to approximately 13 to 16 miles of driving, depending on vehicle efficiency and driving conditions. While this figure may seem modest on a per-project basis, it shows the cumulative energy demand of widespread AI inference, contributing to the broader carbon footprint of the digital economy.

[1] Josh You (2025), "How much energy does ChatGPT use?". Published online at epoch.ai. Retrieved from: 'https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use'

[2] Nidhal Jegham, Marwan Abdelatti, Lassad Elmoubarki, Abdeltawab Hendawi, How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference