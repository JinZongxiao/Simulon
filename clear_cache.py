import os
import shutil

def remove_pycache_folders(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_dir = os.path.join(root, dir_name)
                shutil.rmtree(pycache_dir)
                print(f"Removed {pycache_dir}")

remove_pycache_folders("C:\\Users\\Administrator\\Desktop\\phd_workspace\\code_repo\\Simulon")

