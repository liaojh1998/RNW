from nuscenes.nuscenes import NuScenes
from numba import njit
import numpy as np

def get_file_list(data_dir, video_clip_num):
    """
    given data directory and video clip number, 
    """
    raise NotImplemented

@njit()
def get_intrinsics(sample_data):
    """
    get the intrinsics from the specified sample_data
    """
    calibrated_sensor_token = cam_front_data['calibrated_sensor_token']
    cs_record = nusc.get('calibrated_sensor', calibrated_sensor_token)
    #sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    #print(cs_record)
    return np.array(cs_record["camera_intrinsic"])

def load_scene_list(data_dir):
    """
    load the scene list from the specified data directory
    """
    raise NotImplemented

def parse_scene(scene, out_root_data_dir):
    """
    given a scene, produce the following output to the specified output directory:
    1. create a folder under out_root_data_dir/sequence/ with name of the scene token
    2. get intrinsics and store under root/sequence/videoclipnumber/intrinsic.npy
    3. get the list of image files and store under root/sequence/videoclipnumber/image_list.txt
    4. store the name of image files under root/sequence/videoclipnumber/file_list.txt
    """
    raise NotImplemented
def main():
    DATA_DIR = "data/nuscene/mini"
    OUT_DATA_DIR = "data/parsed_nuscene/mini"
    sensor = "CAM_FRONT"
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_DIR, verbose=True)
    for scene in nusc.scene:
        #print(scene)
        #print(scene["token"])
        first_sample_token = scene["first_sample_token"]
        my_sample = nusc.get('sample', first_sample_token)
        
        #print(my_sample)
        cam_front_sample_data = nusc.get('sample_data', my_sample['data'][sensor])
        #print(cam_front_data)
        calibrated_sensor_token = cam_front_sample_data['calibrated_sensor_token']
        cs_record = nusc.get('calibrated_sensor', calibrated_sensor_token)
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        print(np.array(cs_record["camera_intrinsic"]))

        print(cam_front_sample_data["filename"])
        exit(0)
    pass

if __name__ == "__main__":
    main()