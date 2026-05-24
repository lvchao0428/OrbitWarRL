import kagglehub

# Download latest version
path = kagglehub.dataset_download("bovard/orbit-wars-top10-episodes-2026-05-04")

print("Path to dataset files:", path)
