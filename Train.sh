python -m generativeimage2text.train -p "{'type': 'mytrain', \
'param':{'num_image_with_embedding':8}, \
'args' :{ \
      'num_workers':4, \
      'Pix2Struct':False,\
      'use_dif_lr': True,\
      'wd':0.0001,     \
      'lr':1e-5,    \
      'epoch':9,    \
      'bs':32 ,     \
      'acc_step':8, \
      'pat':2,      \
      'ckpt_path':'/data/cv/poyang/checkpoint/', \
      'load_path':'/data/cv/poyang/checkpoint/1000_2lr_8img_lowWD_lr1e-05_wd0.0001_im8.ckpt',\
      'exp_name' :'1000_2lr_8img_lowWD' \
      }}" 
