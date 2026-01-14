import kagglehub
import os

print("Downloading dataset...")
path = kagglehub.dataset_download("pkdarabi/bone-fracture-detection-computer-vision-project")

print("Path to dataset files:", path)

print("\nDataset Structure:")
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print('{}{}/'.format(indent, os.path.basename(root)))
    subindent = ' ' * 4 * (level + 1)
    # Print only first 5 files to avoid clutter
    for f in files[:5]:
        print('{}{}'.format(subindent, f))
    if len(files) > 5:
        print('{}... ({} files)'.format(subindent, len(files)))
