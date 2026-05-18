<div align=center>

# From Pixels to Semantics: A Novel MLLM-Driven Approach for Explainable Tampered Text Detection

</div>

<div align="center">
  <a href="http://dlvc-lab.net/lianwen/"> <img alt="SCUT DLVC Lab" src="https://img.shields.io/badge/SCUT-DLVC_Lab-B32952?logo=Academia&logoColor=hsl"></a>
  <a href="https://dl.acm.org/doi/abs/10.1145/3746027.3755803"> <img alt="Static Badge" src="https://img.shields.io/badge/ACM_MM-TVSIP-58B822"></a>
<p></p>
</div>

## 💡 Introduction
- We propose TVSIP, a novel explainable framework that combines low-level visual artifact detection with high-level semantic analysis for tampered text verification. It aims to leverage MLLMs to enhance the pixel-level localization ability of expert models while providing detailed and reliable tampering analysis, including image description, tampered text detection, localization, and explanation.
- We present TextDDLE, a meticulously curated benchmark that facilitates both training and evaluation of tampered text analysis capabilities. Created through a systematic pipeline utilizing GPT-4o with expert verification, TextDDLE supports the four fundamental tasks of tampering analysis.
- Extensive experiments demonstrate that semantic clues notably improve model performance and robustness in the TTD task. TVSIP offers strong robustness to image degradation and excellent generalization to unseen scenarios.
---


## 📥 Download

**Dataset**

Since TextDDLE-PT is too large, we keep it separate from the other subsets.

| **Dataset** | **Link** |
|----------|----------|
| TextDDLE-PT | [BaiduYun:p8q6](https://pan.baidu.com/s/1u8CS_3UgE_nVjxxdlu08Gw?pwd=p8q6) |
| TextDDLE wo PT | [BaiduYun:5yre](https://pan.baidu.com/s/11Oh_TlR_BharY8ZGmrhveQ?pwd=5yre) |

**Note:**
- The TextDDLE dataset can only be used for non-commercial research purposes. For scholar or organization who wants to use the TextDDLE dataset, you can apply through either of the following two options:

  **Option A: Apply Online**
  Submit your application through our online platform: 👉 [Apply Here](http://121.41.49.212:9000/)
  
  **Option B: Apply via Email**
  Please first fill in this [Application Form](./application-form/Application-Form-for-Using-TextDDLE.docx) and sign the [Legal Commitment](./application-form/Legal-Commitment.docx) and email them to us ([eelwjin@scut.edu.cn](eelwjin@scut.edu.cn), cc: [eegtxu@mail.scut.edu.cn](eegtxu@mail.scut.edu.cn)). When submitting the application form to us, please list or attached 1-2 of your publications in the recent 6 years to indicate that you (or your team) do research in the related research fields of OCR, image forgery detection and localization, document image processing, and so on.
  
- We will give you the decompression password after your application has been received and approved.
- The original data of the dataset is sourced from public channels such as the Internet, and its copyright shall remain with the original providers. The collated and annotated dataset presented in this case is for non-commercial use only and is currently licensed to universities and research institutions. To apply for the use of this dataset, please fill in the corresponding application form in accordance with the requirements specified on the dataset’s official website. The applicant must be a full-time employee of a university or research institute and is required to sign the application form. For the convenience of review, it is recommended to affix an official seal (a seal of a secondary-level department is acceptable).
- All users must follow all use conditions; otherwise, the authorization will be revoked.

**Model Zoo**

| **Model** | **Checkpoint** |
|-----------|----------------|
| **Locator**                   | [BaiduYun:4ake](https://pan.baidu.com/s/1sNxkZCPvH1KjCZj688Brqg?pwd=4ake) |
| **Pretrained Interpreter**    | [BaiduYun:ibv9](https://pan.baidu.com/s/1AXHiPyyeAx-0CU2YkjURXg?pwd=ibv9) |
| **Fine-tuned Interpreter**         | [BaiduYun:avw5](https://pan.baidu.com/s/13_X9uV5AcmyQ5zPmutiZEg?pwd=avw5) |


**Inference Results of TVSIP**

You can download all inference results of TVSIP from [BaiduYun:j3jb](https://pan.baidu.com/s/1FPVegrWLNho2rzB6IyvSLw?pwd=j3jb).

## ⚒️ Environment

```bash
git clone https://github.com/SCUT-DLVCLab/TVSIP.git
cd TVSIP
conda create --name tvsip --file requirements.txt
conda activate tvsip
```
## 🔥 Training

**Data preparation**
- Download the TextDDLE dataset into the datasets folder.
- Move JSON files in TextDDLE to the data folder.

**For Locator:**
```bash
bash tools/train_locator.sh
```
**For Interpreter:**
```bash
bash tools/train_interpreter_stage1.sh
```
You can also skip the pretraining step and fine-tune directly.
```bash
bash tools/train_interpreter_stage2.sh
```
**Note:** Since visual expert models (i.e., the low-level vision clue branch of Locator in this work) are not the focus of this work, we directly use the results trained by [SegFormer](https://github.com/NVlabs/SegFormer). You can download the inference results of the expert models from [BaiduYun:j3jb](https://pan.baidu.com/s/1FPVegrWLNho2rzB6IyvSLw?pwd=j3jb).

## 🚀 Inference
**For the high-level semantic clue branch of Locator:**
```bash
bash tools/infer_locator.sh
```

**For Interpreter:**
```bash
bash tools/infer_interpreter.sh
```

## 📅 Fusion and Evaluation
**For Locator:**
```bash
bash tools/evaluation_for_locator.sh
```

Also, you can obtain the final fusion results from the Locator

**For Interpreter:**
```bash
bash tools/evaluation_for_interpreter.sh
```

## 📫 Contact

If you have any questions, feel free to contact me at eegtxu@mail.scut.edu.cn.

## 💙 Acknowledgement
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL)
- [SegFormer](https://github.com/NVlabs/SegFormer)
- [DocTamper](https://github.com/qcf-568/DocTamper)
- [RLS26K](https://github.com/QixianHao123/RLS26K_dataset?tab=readme-ov-file)
- [RTM](https://github.com/DrLuo/RTM)
- [DiffUTE](https://github.com/chenhaoxing/DiffUTE)


## 📜 License
The code and dataset should be used and distributed under [ (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) for non-commercial research purposes.

## ⛔️ Copyright
- This repository can only be used for non-commercial research purposes.
- For commercial use, please contact Prof. Lianwen Jin (eelwjin@scut.edu.cn).
- Copyright 2025, [Deep Learning and Vision Computing Lab (DLVC-Lab)](http://www.dlvc-lab.net), South China University of Technology. 

## ✒️ Citation

If you find this paper helpful, please consider giving this repo a ⭐ and citing:
```latex
@inproceedings{xu2025pixels,
  title={From Pixels to Semantics: A Novel MLLM-Driven Approach for Explainable Tampered Text Detection},
  author={Xu, Guitao and Yi, Ziqi and Zhang, Peirong and Cao, Jiahuan and Wu, Shihang and Jin, Lianwen},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={757--766},
  year={2025}
}
```
