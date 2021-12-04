from nuscenes.nuscenes import NuScenes
from numba import njit
import numpy as np
import re
import os

def get_file_list(nusc, first_cam_sample_data):
    """
    given data directory and video clip number, 
    """
    assert first_cam_sample_data["prev"] == "", "first sample data should be the first frame of the video clip"
    cam_sample_data = first_cam_sample_data
    image_name_list = []
    while True:
        # save the image file name
        #print(cam_sample_data["filename"])
        image_name_list.append(cam_sample_data["filename"])
        if cam_sample_data["next"] == "":
            break
        cam_sample_data = nusc.get('sample_data', cam_sample_data["next"])
    
    #return [cam_sample_data["filename"]]
    return image_name_list

def parse_file_list(file_list_file, in_img_data_dir, out_img_data_dir):
    """
    do the following:
    1. put the list of image to outdir/file_list.txt, with name mapping
    2. link the image from the original data directory to outdir/
    """
    p = re.compile("\w+/[\w,\_]+/([\w,\_,.,\-,\+]+)")
    new_img_list_file = []
    for img_fn in file_list_file:
        new_fn = p.match(img_fn).group(1)
        new_img_list_file.append(new_fn)
    # link the image files
    for src, dst in zip(file_list_file, new_img_list_file):
        src = os.path.join(os.getcwd(), in_img_data_dir, src)
        dst = os.path.join(out_img_data_dir, dst)
        assert os.path.exists(src), "the source file does not exist"
        #print(src, dst)
        os.symlink(src, dst)
    # write the file list to the file
    dst = os.path.join(out_img_data_dir, "file_list.txt")
    with open(dst, "w") as f:
        for fn in new_img_list_file:
            f.write(fn + "\n")

def get_intrinsics(nusc, sample_data):
    """
    get the intrinsics from the specified sample_data
    """
    calibrated_sensor_token = sample_data['calibrated_sensor_token']
    cs_record = nusc.get('calibrated_sensor', calibrated_sensor_token)
    #sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    #print(cs_record)
    return np.array(cs_record["camera_intrinsic"])


def parse_scene(nusc, scene, in_root_data_dir, out_root_data_dir, sensor="CAM_FRONT"):
    """
    given a scene, produce the following output to the specified output directory:
    1. create a folder under out_root_data_dir/sequence/ with name of the scene token
    2. get intrinsics and store under root/sequence/videoclipnumber/intrinsic.npy
    3. get the list of image files and store under root/sequence/videoclipnumber/image_list.txt
    4. store the name of image files under root/sequence/videoclipnumber/file_list.txt
    """
    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get('sample', first_sample_token)
    cam_front_sample_data = nusc.get('sample_data', first_sample['data'][sensor])
    # get the intrinsics
    intrinsics = get_intrinsics(nusc, cam_front_sample_data)

    # get the list of image files
    image_list = get_file_list(nusc, cam_front_sample_data)

    in_img_data_dir = in_root_data_dir
    out_img_data_dir = os.path.join(out_root_data_dir, "sequences", scene["token"])
    
    is_exist = os.path.exists(out_img_data_dir)
    if not is_exist:  
        # Create a new directory because it does not exist 
        os.makedirs(out_img_data_dir)
        print("Creating {}".format(out_img_data_dir))
    # stores the intrinsics
    np.save(os.path.join(out_img_data_dir, "intrinsic.npy"), intrinsics)
    # parse the image list
    parse_file_list(image_list, in_img_data_dir, out_img_data_dir)


    #calibrated_sensor_token = cam_front_sample_data['calibrated_sensor_token']
    #cs_record = nusc.get('calibrated_sensor', calibrated_sensor_token)
    #sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    #print(np.array(cs_record["camera_intrinsic"]))

    #print(cam_front_sample_data["filename"])

def test():
    DATA_DIR = "data/nuscene/mini"
    OUT_DATA_DIR = "data/parsed_nuscene/mini"
    sensor = "CAM_FRONT"
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_DIR, verbose=True)
    for scene in nusc.scene:
        #print(scene)
        #print(scene["token"])
        first_sample_token = scene["first_sample_token"]
        first_sample = nusc.get('sample', first_sample_token)
        print(first_sample["data"].keys())
        exit(0)
        #print(first_sample)
        cam_front_sample_data = nusc.get('sample_data', first_sample['data'][sensor])
        #print(cam_front_data)
        calibrated_sensor_token = cam_front_sample_data['calibrated_sensor_token']
        cs_record = nusc.get('calibrated_sensor', calibrated_sensor_token)
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        print(np.array(cs_record["camera_intrinsic"]))

        print(cam_front_sample_data["filename"])
        exit(0)
    pass
    

def main():
    DATA_DIR = "data/nuscene/mini"
    OUT_DATA_DIR = "data/parsed_nuscene/mini"
    sensor = "CAM_FRONT"
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_DIR, verbose=True)
    for scene in nusc.scene:
        parse_scene(nusc, scene, DATA_DIR, OUT_DATA_DIR, sensor)


if __name__ == "__main__":
    #test()
    main()