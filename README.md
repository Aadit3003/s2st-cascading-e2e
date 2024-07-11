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

Note: The model files (for the TTS model, Vocoder, S2T model etc.) are not included due to their size, however all config files are in the respective directories.

## Run Commands

Environment Setup
Inference the Models
Fine-tune the Cascaded model
Run the Demo



## TO DO

1. Rename the files (DONE)
2. Write File description docstrings (DONE)
3. Cleanup Code - Main Function, paths, function names etc. (DONE)
4. Write Function description docstrings
5. Finish writing Readme!
