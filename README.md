# Learning without Forgetting for Vision-Language Models



<div align="center">
    <div>
        <a href='http://www.lamda.nju.edu.cn/zhoudw' target='_blank'>Da-Wei Zhou</a><sup>1</sup>&emsp;
        <a href='https://zhangyuanhan-ai.github.io/' target='_blank'>Yuanhan Zhang</a><sup>2</sup>&emsp;
        Yan Wang<sup>1</sup>&emsp;
        <a href='https://jingyinju.github.io/' target='_blank'>Jingyi Ning</a><sup>1</sup>&emsp;
        <a href='http://www.lamda.nju.edu.cn/yehj' target='_blank'>Han-Jia Ye</a><sup>1</sup>&emsp;
        <a href='http://www.lamda.nju.edu.cn/zhandc' target='_blank'>De-Chuan Zhan</a><sup>1</sup>&emsp;
        <a href='http://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a><sup>2</sup>
    </div>
    <div>
    <sup>1</sup>School of Artificial Intelligence, State Key Laboratory for Novel Software Technology, Nanjing University<br>
    <sup>2</sup>S-Lab, Nanyang Technological University
    </div>
</div>




<div align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=zhoudw-zdw.Proof&left_color=yellow&right_color=purple" alt="visitors">
  <a href="https://arxiv.org/abs/2305.19270">
    <img src="https://img.shields.io/badge/TPAMI2025-red" alt="arXiv">
  </a>

  
</div>



The code repository for "[Learning without Forgetting for Vision-Language Models](https://arxiv.org/abs/2305.19270) (**TPAMI 2025**)"  in PyTorch.  If you use any content of this repo for your work, please cite the following bib entry: 

```bibtex
@article{zhou2025learning,
  title={Learning without Forgetting for Vision-Language Models},
  author={Da-Wei Zhou and Yuanhan Zhang and Yan Wang and Jingyi Ning and Han-Jia Ye and De-Chuan Zhan and Ziwei Liu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}
```



# 📢 **Updates**

[02/2025] Accepted to TPAMI.

[10/2024] Code has been released.

[05/2023] [arXiv](https://arxiv.org/abs/2305.19270v1) paper has been released.


# 📝 Introduction

Class-incremental learning (CIL) aims to adapt to emerging new classes without forgetting old ones. Traditional CIL models are trained from scratch to continually acquire knowledge as data evolves.While traditional CIL methods focus on visual information to grasp core features, recent advances in Vision-Language Models (VLM) have shown promising capabilities in learning generalizable representations with the aid of textual information. However, when continually trained with new classes, VLMs often suffer from catastrophic forgetting of former knowledge. Applying VLMs to CIL poses two major challenges: 1) how to adapt the model without forgetting; and 2) how to make full use of the multi-modal information. To this end, we propose PROjectiOn Fusion (PROOF) that enables VLMs to learn without forgetting. To handle the first challenge, we propose training task-specific projections based on the frozen image/text encoders. When facing new tasks, new projections are expanded, and former projections are fixed, alleviating the forgetting of old concepts. For the second challenge,  we propose the fusion module to better utilize the cross-modality information. By jointly adjusting visual and textual features, the model can capture better task-specific semantic information that facilitates recognition. Extensive experiments on nine benchmark datasets with various continual learning scenarios and various VLMs validate that PROOF achieves state-of-the-art performance.

<div align="center">
<img src="resources/img.png" width="95%">
</div>

## 🔧 Requirements

**Environment**

1 [torch 1.11.0](https://github.com/pytorch/pytorch)

2 [torchvision 0.12.0](https://github.com/pytorch/vision)

3 [open-clip 2.17.1](https://github.com/mlfoundations/open_clip/releases/tag/v2.17.1)

**Dataset**

We provide the processed datasets as follows:

- **CIFAR100**: will be automatically downloaded by the code.
- **CUB200**: Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or OneDrive [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc)
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
- **ObjectNet**: Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy) You can also refer to the [filelist](https://drive.google.com/file/d/147Mta-HcENF6IhZ8dvPnZ93Romcie7T6/view?usp=sharing) and processing [code](https://github.com/zhoudw-zdw/RevisitingCIL/issues/2#issuecomment-2280462493) if the file is too large to download.
- **Cars**: Google Drive: [link](https://drive.google.com/file/d/1D8ReAuOPenWi6SMNUrOZhbm6ViyhDHbL/view?usp=sharing  ) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EbT1XAstg51Mpy82uHM0D2EBJLrtzmr_V64jeBRjqyyTnQ?e=h6g1rM)
- **UCF**: Google Drive: [link](https://drive.google.com/file/d/1Ng4w310_VDqpKbc7eYaumXTOiDxI02Wc/view?usp=sharing) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EU2qHQXjASdLh1jIl6ihZmcB6G2KvqmSw-sTlZKDE6xPbg?e=7ezvTr)
- **Aircraft**: Google Drive: [link](https://drive.google.com/file/d/1xI5r1fU0d6Nff51HuOo5w-e4sGEP46Z2/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/ETVliZnmPY9AvZZgcFFJ6jMB2c7TRvcq7-gso2Aqvdl_VQ?e=pWXqdP)
- **Food**: Google Drive: [link](https://drive.google.com/file/d/1rupzXpwrbxki4l-RVmsRawhz1Cm0lDY5/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/Eb4xfptD4L5Egus-SiYxrIcBDH1VewLGp4kzyACGF_Na_w?e=duA3Ia)
- **SUN**: OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EcQq1-1pFulKstYtdknB4O8BGo0hnlDRarAwB4wFEgkx0Q?e=YZ0xYV)
- **TV100**: [link](https://tv-100.github.io/)

These subsets are sampled from the original datasets. Please note that I do not have the right to distribute these datasets. If the distribution violates the license, I shall provide the filenames instead.

You need to modify the path of the datasets in `./utils/data.py` according to your own path. 

## 💡 Running scripts

To prepare your JSON files, refer to the settings in the `exps` folder and run the following command. All main experiments from the paper are already provided in the `exps` folder, you can simply execute them to reproduce the results found in the `logs` folder.

```
python main.py --config ./exps/[configname].json
```

## 🎈 Acknowledgement

This repo is based on [CIL_Survey](https://github.com/zhoudw-zdw/CIL_Survey) and [PyCIL](https://github.com/G-U-N/PyCIL). 

## 💭 Correspondence

If you have any questions, please  contact me via [email](mailto:zhoudw@lamda.nju.edu.cn) or open an [issue](https://github.com/zhoudw-zdw/Proof/issues/new).

