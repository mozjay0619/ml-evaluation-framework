import zipfile
import os
import errno


def unzip_dir(dir_path, dirname):
    
    path, file = os.path.split(dir_path)
    
    new_dir_path = os.path.join(path, dirname)
    
    try:
        os.makedirs(new_dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    with zipfile.ZipFile(dir_path, 'r') as f:
        f.extractall(new_dir_path)

    return new_dir_path
