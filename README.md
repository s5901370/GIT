# Introduction 
This repo presents some example codes to reproduce some results in
[GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100).

# Installation
- Install [azfuse](https://github.com/microsoft/azfuse). The tool is used to
  automatically download the data. The configuration of
  AzFuse has already been in this repo.

- Download the source code by
  ```shell
  git clone https://github.com/s5901370/GIT.git
  cd GenerativeImage2Text
  ```

- Install the package
  ```shell
  pip install -r requirements.txt
  python setup.py build develop
  sudo apt install openjdk-11-jdk
  ```
# Before Starting

# Inference
- Change the **annotation_files** and **data_path** in the [generativeimage2text/train.py](https://github.com/s5901370/GIT/blob/6a346b502253761a71776f5236e60226ba3d5cc3/generativeimage2text/train.py#L538)
- Run the bash file "Train.sh" and replace the 'type' with 'myinfer'.
- Please notice the 'load_path', and the correct settings of 'num_image_with_embedding', 'Pix2Struct' and 'use_dif_lr'.
  ```
   python -m generativeimage2text.train -p "{'type': 'myinfer',
  'param':{'num_image_with_embedding':8}, 
  'args' :{
      'num_workers':4, 
      'Pix2Struct':False,
      'use_dif_lr': True,
      'wd':0.0001,     
      'lr':1e-5,    
      'epoch':1,    
      'bs':32 ,     
      'acc_step':8, 
      'pat':2,      
      'ckpt_path':'/data/cv/poyang/checkpoint/', 
      'load_path':'/data/cv/poyang/checkpoint/1000_2lr_8img_lowWD_lr1e-05_wd0.0001_im8.ckpt',
      'exp_name' :'1000_2lr_8img_lowWD'
      }}" 
  ```
# Train
- Change the **annotation_files** and **data_path** in the [generativeimage2text/train.py](https://github.com/s5901370/GIT/blob/6a346b502253761a71776f5236e60226ba3d5cc3/generativeimage2text/train.py#L348)
- If you don't need to load pretrained model weight, please delete 'load_path'.
  ```
   python -m generativeimage2text.train -p "{'type': 'mytrain',
  'param':{'num_image_with_embedding':8}, 
  'args' :{
      'num_workers':4, 
      'Pix2Struct':False,
      'use_dif_lr': True,
      'wd':0.0001,     
      'lr':1e-5,    
      'epoch':3,    
      'bs':32 ,     
      'acc_step':8, 
      'pat':2,      
      'ckpt_path':'/data/cv/poyang/checkpoint/', 
      'exp_name' :'1000_2lr_8img_lowWD'
      }}" 
  ```
# Citation
Please consider to cite the following reference if it helps.
```text
@article{wang2022git,
  title={GIT: A Generative Image-to-text Transformer for Vision and Language},
  author={Wang, Jianfeng and Yang, Zhengyuan and Hu, Xiaowei and Li, Linjie and Lin, Kevin and Gan, Zhe and Liu, Zicheng and Liu, Ce and Wang, Lijuan},
  journal={arXiv preprint arXiv:2205.14100},
  year={2022}
}
```
# Acknowledgement
Part of the code is based on
[transformers](https://github.com/huggingface/transformers),
[clip](https://github.com/openai/CLIP),
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark),
[oscar](https://github.com/microsoft/Oscar),
[virtex](https://github.com/kdexd/virtex).


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
