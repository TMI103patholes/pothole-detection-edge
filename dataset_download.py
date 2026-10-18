import os
import traceback
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

try:
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("yeeun-kim-fyvoj").project("pothole-vhmow")
    version = project.version(18)

    dataset = version.download("yolov8", location="data/dataset")

    print("Reported location:", dataset.location)
    print("Absolute path:", os.path.abspath(dataset.location))
    print("Exists?", os.path.exists(dataset.location))

    if os.path.exists(dataset.location):
        print("Contents:", os.listdir(dataset.location))
    else:
        print("Folder does not actually exist on disk.")

except Exception as e:
    print("ERROR OCCURRED:")
    traceback.print_exc()