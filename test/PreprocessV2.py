import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#hide error tf message 
import tensorflow as tf
from PIL import Image
import pickle
import jsonlines
from tqdm import tqdm
def _decode_image(example):
    """Decodes image from example and reshapes.
    
    Args:
    example: Example which contains encoded image.
    image_height: The height of the raw image.
    image_width: The width of the raw image.
    image_channels: The number of channels in the raw image.

    Returns:
    Decoded and reshaped image tensor. also transform to pytorch tensor or numpy array
    """
    image = tf.io.decode_raw(
        example.features.feature['image/encoded'].bytes_list.value[0],
        out_type=tf.uint8,
    )
    image_height = example.features.feature['image/height'].int64_list.value[0]
    image_width = example.features.feature['image/width'].int64_list.value[0]
    image_channels = example.features.feature['image/channels'].int64_list.value[0]

    height = tf.cast(image_height, tf.int32)
    width = tf.cast(image_width, tf.int32)
    n_channels = tf.cast(image_channels, tf.int32)

    return tf.reshape(image, (height, width, n_channels)).numpy()
def find_miss_file(dataset,name):
    # return a set consists of 'episode id' missing files
    sum = 0
    count = {}
    total = {}
    miss_img_set = set() #ret.add  len(ret)
    for d in tqdm(dataset):

        # get episode
        sum += 1
        ex = tf.train.Example()
        ex.ParseFromString(d)
        ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        step_id = ex.features.feature['step_id'].int64_list.value[0]
        ep_length = ex.features.feature['episode_length'].int64_list.value[0]

        # print(f'id = {ep_id},sum = {sum:03d}, step_id = {step_id:02d}, length = {ep_length}')
        # if ep_id == '6160402803482314990' or ep_id == '2729424677228029046':
        #     print(f'sum = {sum}, step_id = {step_id}, length = {ep_length}')

        if ep_id not in count:
            count[ep_id] = 1
            total[ep_id] = ep_length
        else:
            count[ep_id] += 1
    print(f"sum = {sum}")
    for k,v in count.items():
        if v == total[k]:
            continue
        else:
            miss_img_set.add(k)
            # print(k)
            # print('count = ',v,', length = ',total[k])
    print('mis img epi = ',len(miss_img_set))
    with open(name, 'wb') as handle:
        pickle.dump(miss_img_set, handle, protocol=pickle.HIGHEST_PROTOCOL)


def print_img(dataset,des,miss_set,jsonl_name,category):
    writer = jsonlines.open(jsonl_name, mode='a')
    for d in tqdm(dataset):
        ex = tf.train.Example()
        ex.ParseFromString(d)
        ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        if ep_id in miss_set:
            # print(ep_id)
            continue
        step_id = ex.features.feature['step_id'].int64_list.value[0]
        ep_length = ex.features.feature['episode_length'].int64_list.value[0]
        # only write one time
        if step_id == 0:
            goal_info = ex.features.feature['goal_info'].bytes_list.value[0].decode('utf-8')
            writer.write({"category":category,"id":ep_id,"goal_info":goal_info,"episode_length":ep_length})
        # Image.fromarray(_decode_image(ex)).save(f"{des}{ep_id}-{step_id}-{ep_length}.jpg")
    writer.close()

def findWH(example):

    image = tf.io.decode_raw(
        example.features.feature['image/encoded'].bytes_list.value[0],
        out_type=tf.uint8,
    )
    image_height = example.features.feature['image/height'].int64_list.value[0]
    image_width = example.features.feature['image/width'].int64_list.value[0]
    return (image_width,image_height)

def main():
    # N = 1000
    # dir = ['general','install','web_shopping']
    category = 'WebShopping'
    # dir = ['General','GoogleApps','Install','WebShopping']
    dir = ['General','Install','WebShopping']
    # for category in dir:
    # for category in dir:
    data_src = f'/data/poyang/android-in-the-wild/{category}/'
    data_des = f'/data/poyang/no-miss-AITW/{category}/'
    # jsonl_name = f'no_miss_1000_train.jsonl'
    jsonl_name = f'no_miss_{category}_train.jsonl'
    pkl_name = f'AiTW_Miss_Img_ID_{category}.pickle'
    # pkl_name = f'AiTW_Miss_Img_ID_1000.pickle'

    # pkl_name = f'AiTW_Miss_Img_ID_{category}.pickle'
    # file0 = 'google_apps-00000-of-08688' #88
    # file1 = 'google_apps-00001-of-08688' #205
    files_in_directory = os.listdir(data_src)
    # print(len(files_in_directory))#8687
    # print(files_in_directory[:105])
    # train_files = [data_src+file0]
    # train_files = [data_src+file0,data_src+file1]
    train_files = [(data_src + i) for i in files_in_directory]

    # last time :105
    # last time 105:1000
    # next time 1000:
    # train_files = [(data_src + i) for i in files_in_directory[:1000]]
    raw_dataset = tf.data.TFRecordDataset(train_files, compression_type='GZIP').as_numpy_iterator()

    # record id who misses images
    # find_miss_file(raw_dataset,pkl_name)

    # load pickle
    with open(pkl_name, 'rb') as handle:
        loaded_set = pickle.load(handle)
    print_img(raw_dataset,data_des,loaded_set,jsonl_name,category)
    # print(loaded_set)
        
    
    

def countWH():
    dir = ['General','GoogleApps','Install','WebShopping']
    count = {} #ret.add  len(ret)
    for category in dir:
        count_in = {}
        data_src = f'/data/poyang/android-in-the-wild/{category}/'
        files_in_directory = os.listdir(data_src)
        train_files = [(data_src + i) for i in files_in_directory]
        raw_dataset = tf.data.TFRecordDataset(train_files, compression_type='GZIP').as_numpy_iterator()
        for i,d in enumerate(tqdm(raw_dataset)):
            ex = tf.train.Example()
            ex.ParseFromString(d)
            ret = findWH(ex)
            if ret not in count:
                count[ret] = 1
            else:
                count[ret] += 1
            if ret not in count_in:
                count_in[ret] = 1
            else:
                count_in[ret] += 1
        print(category)
        print(count_in)
    print('total')
    print(count)
    
def check_dup():
     with jsonlines.open("no_miss_GoogleApps_train.jsonl") as reader:
        miss = set()
        for lines in reader:
            id = lines['id']
            if id not in miss:
                miss.add(id)
            else:
                print(f'find duplicate {id}')
    # find duplicate 3883174507157444369
    # find duplicate 10914442784645576315

def countImg():
    with jsonlines.open("no_miss_GoogleApps_train.jsonl") as reader:
        miss = dict()
        total = 0
        for lines in reader:
            total += 1
            num = lines['episode_length']
            num = (num if num <= 30 else 30)
            if num not in miss:
                miss[num] = 1
            else:
                miss[num] += 1
        for i in sorted(miss.keys()):
            print(f"{i}: {miss[i]}",end=',')
    print(f"total = {total}")
'''
{1: 3107,2: 9404,3: 27182,4: 49769,5: 68496,6: 74774,7: 69541,8: 57030,9: 43365,10: 33303,
11: 24313,12: 18292,13: 13403,14: 9750,15: 7156,16: 5310,17: 4164,18: 3192,19: 2589,20: 2029,
21: 1741,22: 1451,23: 1206,24: 976,25: 811,26: 666,27: 493,28: 510,29: 420,30: 2673}
'''
            
    
if __name__ == '__main__':
    main()
    # countWH()
    # check_dup()
    # countImg()





