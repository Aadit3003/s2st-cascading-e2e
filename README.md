# A comparison of E2E and Cascading Speech-to-Speech translation systems (CVSS-C ES-EN)
aa

This project was completed under the guidance of Prof. Shinji Watanabe, at LTI, CMU. For more details refer to the [project report](https://github.com/Aadit3003/s2st-cascading-e2e/blob/8a1be02494e6ecac6c0db413026a399bdf916a9b/report.pdf).


## Results
| Model           | ASR-BLEU      | BP            | HRR            | COMET          | METEOR         | BLASER2.0      |
|-----------------|---------------|---------------|----------------|----------------|----------------|----------------|
| E2E-oob         | 14.901        | 0.82          | 0.928          | 0.538          | 0.283          | 3.188          |
| casc-oob        | **17.692**        | **0.88**          | **0.975**          | **0.619**          | **0.338**         | **3.604**          |
| casc-ft-best    | 15.062        | 0.709         | 0.785          | 0.599          | 0.323          | 3.435          |
| casc-ft-1-epoch | 14.930        | 0.705         | 0.784          | 0.593          | 0.318          | 3.386          |
| casc-ft-5-epoch | 14.383        | 0.702         | 0.784          | 0.601          | 0.314          | 3.428          |
| casc-ft-low-lr  | 13.031        | 0.636         | 0.722          | 0.570          | 0.281          | 3.298          |

aa

## Directory Structure
* cvss-c_en_wavegan_hubert_vocoder -
* dev_dataset -
* espnet_recipe_scripts -
* results -
* tts_multi_speaker_model - 
* utils -
 * macro_average_results.py -
 * sampling_rate_converter.py - 

  expanded_translation_metrics.py -
  finetune_s2t.py -
  forward_feed_cascaded_finetuned_oob.py - 
  forward_feed_e2e.py - 
  live_s2st_demonstration.py - 

tts_config.yaml
lora_config.yaml

environment.txt
report.pdf

Note: The model files (for the TTS model, Vocoder, S2T model etc.) are not included due to their size, however all config files are in the respective directories.

## Run Commands

* Environment Setup:
  * Since this project extensively uses espnet recipes, refer to the following [espnet installation](https://github.com/espnet/espnet) instructions.
* Fine-tune the Cascaded model on CoVoST 2 data:
  * Download the [CoVoST](https://huggingface.co/datasets/facebook/covost2/) 2 es_en dataset (edit the corresponding path variable in the script).
  * ```python finetune_s2t.py```
* Inference the Models on the dev dataset and calculate all metrics
  * Download the appropriate audio files in the CVSS-C es-en dev dataset from the [CommonVoice](https://commonvoice.mozilla.org/en/datasets) release version 4.
  * End-to-end model: ```python forward_feed_e2e.py```
  * Cascaded oob model: ```python forward_feed_cascaded_finetuned_oob.py --inference_mode=oob```
  * Cascaded finetuned model: ```python forward_feed_cascaded_finetuned_oob.py --inference_mode=finetuned```
* Run the Demonstration:
  * The demo requires the finetuning step to have been completed prior.
  * Select a single file from the CVSS-C es-en dataset to inference and change the 'demo_sample_filename' variable.
  * Run the demo: ```python live_s2t_demonstration.py```
