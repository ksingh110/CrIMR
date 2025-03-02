import os
output_path = "E:\\cry1controlsequencesencoded.npz"
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)