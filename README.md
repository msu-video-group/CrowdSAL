# CrowdSAL: Video Saliency Dataset and Benchmark

<img width="1920" height="471" alt="Teaser_scaled" src="https://github.com/user-attachments/assets/3c63d7bb-c5cf-4d22-9d59-ba6dced25835" />


## Dataset
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/ANDRYHA/CrowdSAL)
[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1daH-14w_vHLc9OuGQ_RU0HgUv_Wc3G0o?usp=sharing)


**CrowdSAL** is the largest video saliency dataset with the following key features:
* Large scale: **5000** videos with mean **18.4s** duration, **2.7M+** frames;
* Mouse fixations from **>19000** observers (**>75** per video);
* **Audio** track saved and played to observers;
* High resolution: all streams are **FullHD**;
* Diverse content from **YouTube, Shorts, Vimeo**;
* License: **CC-BY**;

### File Structure

1) `Train/Test` folders — dataset splits, ids 0001-3000 are from Train, 3001-5000 from Test subset;
  
2) `Videos` — 5000 mp4 FullHD, 30 FPS videos with audio streams;

3) `Saliency` — 5000 mp4 almost losslessly (crf 0, 10bit, min-max normalized) compressed continuous saliency maps videos;

4) `Fixations` — 5000 json files with per-frame fixation coordinates, from which saliency maps were obtained;

5) `metadata.jsonl` — meta information about each video (e.g. license, source URL, etc.)


## Benchmark Evaluation

### Environment Setup

```
conda create -n saliency python=3.10.19
conda activate saliency
pip install numpy==2.2.6 opencv-python-headless==4.12.0.88 tqdm==4.67.1
conda install ffmpeg=4.4.2 -c conda-forge
```
### Run Evaluation
Usage example:

1) Check that your predictions match the structure and names of the Test dataset subset;
2) Install all dependencies from Environment Setup;
3) Download and extract all CrowdSAL files from the dataset page;
4) Run `python bench.py` with flags:
* `--model_video_predictions` — folder with predicted saliency videos
* `--model_extracted_frames` — folder to store prediction frames (should not exist at launch time)
* `--gt_video_predictions` — folder from dataset page with gt saliency videos
* `--gt_extracted_frames` — folder to store ground-truth frames (should not exist at launch time)
* `--gt_fixations_path` — folder from dataset page with gt saliency fixations
* `--mode` — Train/Test subsets split
* `--results_json` — path to the output results json
5) The result you get will be available following `results_json` path.
