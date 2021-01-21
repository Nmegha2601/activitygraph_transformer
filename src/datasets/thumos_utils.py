import os,sys

def thumos_name_dicts():

    src_dict = {'validation':1, 'test':2}

    return src_dict

def getVideoId(video_name):
    src_name = video_name.split('_')[1]
    video_id = int(video_name.split('_')[2])

    src_dict = thumos_name_dicts()    
    video_id = [src_dict[src_name], video_id]

    #print("video {} person {} camera {} recipe {} : id = {}".format(video_name,person_name,src_name,recipe_name,video_id))
    
    return video_id

def getVideoName(video_id):
    src_dict = thumos_name_dicts()  
    src_dict = {v:k for k,v in src_dict.items()}

    video_name = "video" + "_" + src_dict[video_id[0]] + "_" + str(video_id[1]).zfill(7)

    #print("video {} person {} camera {} recipe {} : id = {}".format(video_id,person_dict[video_id[0]],src_dict[video_id[1]],recipe_dict[video_id[2]],video_name))
    
    return video_name

    

if __name__ == "__main__":
    vid = getVideoId(sys.argv[1])
    reverse = getVideoName(vid)
    print(sys.argv[1], vid, reverse)

