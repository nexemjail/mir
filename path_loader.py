import os
from random import shuffle


def get_audio_files_from_dir(folder):
    path_list = []
    for filename in os.listdir(folder):
        if filename.endswith('.au'):
            path_list.append(os.path.join(folder,filename))
    return path_list


def scan_folders_for_au_files(root_folder = '/media/files/musicsamples/genres'):
    folder_name_file_names_dict = {}
    for file_name in os.listdir(root_folder):
        full_path = os.path.join(root_folder, file_name)
        if os.path.isdir(full_path):
            files = get_audio_files_from_dir(full_path)
            shuffle(files)
            folder_name_file_names_dict[file_name] = files
    return folder_name_file_names_dict