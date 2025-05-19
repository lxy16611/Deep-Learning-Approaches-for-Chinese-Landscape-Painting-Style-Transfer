
## Abstract

*Our project explores deep learning techniques to translate modern landscape photographs into images that emulate the unique aesthetics of Chinese landscape painting. We evaluate multiple generative models, including [Neural Style Transfer](https://arxiv.org/abs/1508.06576), [CycleGAN](https://arxiv.org/abs/1703.10593), [Paint-CUT](https://www.mdpi.com/2076-3417/14/4/1430), [CCLAP-SA](https://github.com/Robin-WZQ/CCLAP), and the newly adapted [CycleGAN-Turbo](https://github.com/GaParmar/img2img-turbo), with a focus on capturing brush texture, compositional layout, and structural fidelity. Through quantitative metrics such as [Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500), [DINO-struct scores](https://arxiv.org/abs/2104.14294), and [LPIPS distance](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html), we demonstrate in visual Turing tests that CycleGAN-Turbo achieves superior results in both style fidelity and content preservation. Furthermore, we show that CycleGAN-Turbo maintains high-quality performance even when trained on small datasets, owing to its diffusion-based backbone ([SD-Turbo](https://arxiv.org/abs/2403.02360)) and fine-tuning with [LoRA adapters](https://arxiv.org/abs/2106.09685). Our findings contribute to the broader effort of integrating traditional art with modern AI, offering tools to preserve and reinterpret Chinese painting in the digital age.*


### Result Compare bewteen different approach 
![result_new](https://github.com/user-attachments/assets/454ded4e-d3b8-4968-a3c9-9d1fd572a578)


## Selected References (with Links)

- [Neural Style Transfer – Gatys et al. (2015)](https://arxiv.org/abs/1508.06576)
- [CycleGAN – Zhu et al. (2017)](https://arxiv.org/abs/1703.10593)
- [Paint-CUT – Sun et al. (2024, Applied Sciences)](https://www.mdpi.com/2076-3417/14/4/1430)
- [CCLAP – Wang et al. (GitHub)](https://github.com/Robin-WZQ/CCLAP)
- [CycleGAN-Turbo – Parmar et al. (GitHub)](https://github.com/GaParmar/img2img-turbo)
- [FID – Heusel et al. (2017)](https://arxiv.org/abs/1706.08500)
- [DINO – Caron et al. (2021)](https://arxiv.org/abs/2104.14294)
- [LPIPS – Zhang et al. (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html)
- [LoRA – Hu et al. (2021)](https://arxiv.org/abs/2106.09685)
- [SD-Turbo backbone – Chen et al. (2024)](https://arxiv.org/abs/2403.02360)
- [Landscape Dataset – Koishi70 (GitHub)](https://github.com/koishi70/Landscape-Dataset)
- [Neural Style Transfer Implementation – Nazia Nafis (GitHub)](https://github.com/nazianafis/Neural-Style-Transfer)
- [Paint-CUT Implementation – haoyuelee (GitHub)](https://github.com/haoyuelee/Paint-CUT)
- [Chinese Painting Generator – Shi Rui (GitHub)](https://github.com/ThreeSRR/Chinese-Painting-Generator)
