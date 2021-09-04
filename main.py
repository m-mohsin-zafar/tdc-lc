import os
import csv
from pipeline.inference import Inference
from tqdm import tqdm
from PIL import Image


# Constants
OUTPUT_DIR = './output/'
DATA_DIR = os.path.join(os.getcwd(), 'data')

# Pipeline Object for Inference
_pipeline = Inference()

with open(os.path.join(OUTPUT_DIR, 'result_count.csv'), 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['file_name', 'count'])
    for f in tqdm(os.listdir(DATA_DIR)[:21]):
        if f.endswith('.png'):
            fp = os.path.join(DATA_DIR, f)
            count, viz = _pipeline.infer(fp)
            if count > 0:
                img = Image.fromarray(viz)
            else:
                img = Image.new('RGB', (256, 256), (0, 0, 0))
            img.save(os.path.join(OUTPUT_DIR, 'viz', f))
            writer.writerow([f, count])