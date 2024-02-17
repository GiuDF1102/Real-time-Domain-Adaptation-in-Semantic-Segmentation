# Real-time Domain Adaptation in Semantic Segmentation
by Antonio Ferrigno, Giulia Di Fede, Vittorio Di Giorgio

In this work, we tackle the challenging task of real-time domain adaptation in semantic segmentation. We experiment on a novel and efficient architecture, the [Short-Term Dense Concatenate (STDC)](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Rethinking_BiSeNet_for_Real-Time_Semantic_Segmentation_CVPR_2021_paper.pdf) network, for semantic segmentation. We combine this with adversarial learning to align the feature distributions of the source and target domains, as specified in [Learning to Adapt Structured Output Space for Semantic Segmentation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tsai_Learning_to_Adapt_CVPR_2018_paper.pdf).

We also explore several extensions to improve the performance and efficiency of the domain adaptation process. These include:
- [Depthwise Separable Convolution](https://arxiv.org/pdf/1610.02357.pdf)
- [Fourier Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf)
- [Self-supervised Training with Multi-band Transfer](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf).

We conduct extensive experiments on two datasets, GTA V and Cityscapes. Our results demonstrate significant advancements in reducing the domain gap and enhancing the segmentation accuracy in real-time scenarios.
