import os
import subprocess


def download_model(url, dest):
    try:
        output = subprocess.check_output(["pget", "-x", url, "/src/tmp"])
    except subprocess.CalledProcessError as e:
        # If download fails, clean up and re-raise exception
        print(e.output)
        raise e
    
    os.rename("/src/tmp/", dest)