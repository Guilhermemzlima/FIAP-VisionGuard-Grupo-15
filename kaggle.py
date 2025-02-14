import kagglehub

# Download latest version
path = kagglehub.dataset_download("kruthisb999/guns-and-knifes-detection-in-cctv-videos")

print("Path to dataset files:", path)