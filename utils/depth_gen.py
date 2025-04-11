import cv2
import numpy as np
import os
from tqdm import tqdm

# --- Configuration ---
# Base directories (without sequence number)
BASE_DATA_DIR = "/home/krkavinda/Datasets/KITTI_raw/kitti_data/scan"  # Base directory for image data
BASE_CALIB_DIR = "/home/krkavinda/Datasets/KITTI_raw/kitti_data/calib"  # Base directory for calibration files

# Sequence range (00 to 10 for KITTI odometry dataset)
SEQUENCES = [f"{i:02d}" for i in range(11)]  # ['00', '01', ..., '10']

# --- Calibration Parsing ---
def load_calibration(calib_file):
    """Load focal length and baseline from KITTI calibration file."""
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    # Parse P2 (left camera) and P3 (right camera) projection matrices
    focal_length = None
    baseline = None
    for line in lines:
        if line.startswith('P2:'):
            P2 = list(map(float, line.split()[1:]))
            focal_length = P2[0]  # fx from P2
        if line.startswith('P3:'):
            P3 = list(map(float, line.split()[1:]))
            baseline = abs(P3[3] / P3[0])  # Baseline = |tx / fx|, tx is in P3[3]
    
    if focal_length is None or baseline is None:
        raise ValueError("Could not parse focal length or baseline from calibration file.")
    
    return focal_length, baseline

# --- Stereo Depth Map Generator ---
class StereoDepthGenerator:
    def __init__(self, focal_length, baseline):
        """Initialize with camera parameters."""
        self.focal_length = focal_length
        self.baseline = baseline
        
        # Initialize StereoSGBM
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,  # Must be divisible by 16
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def compute_disparity(self, left_img, right_img):
        """Compute disparity map from stereo pair."""
        disparity = self.stereo.compute(left_img, right_img).astype(np.float32) / 16.0
        return disparity

    def disparity_to_depth(self, disparity):
        """Convert disparity to depth."""
        depth = (self.focal_length * self.baseline) / (disparity + 1e-6)  # Avoid division by zero
        depth = np.clip(depth, 0.5, 80)  # Clip to reasonable range for KITTI (0.5m to 80m)
        return depth

    def process_pair(self, left_img_path, right_img_path, output_path):
        """Process a single stereo pair and save the depth map."""
        # Load images
        left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
        
        if left_img is None or right_img is None:
            print(f"Error loading images: {left_img_path} or {right_img_path}")
            return
        
        # Compute disparity and depth
        disparity = self.compute_disparity(left_img, right_img)
        depth = self.disparity_to_depth(disparity)
        
        # Save depth map as NumPy array
        np.save(output_path, depth)
        print(f"Saved depth map to {output_path}")

# --- Main Execution ---
def main():
    # Process each sequence
    for seq in SEQUENCES:
        print(f"\nProcessing sequence {seq}...")
        
        # Construct paths for this sequence
        LEFT_IMG_DIR = os.path.join(BASE_DATA_DIR, seq, "image_02")
        RIGHT_IMG_DIR = os.path.join(BASE_DATA_DIR, seq, "image_03")
        CALIB_FILE = os.path.join(BASE_CALIB_DIR, f"{seq}.txt")
        OUTPUT_DIR = os.path.join(BASE_DATA_DIR, seq, "depth")

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Load calibration data
        try:
            focal_length, baseline = load_calibration(CALIB_FILE)
            print(f"Loaded calibration for sequence {seq}: focal_length={focal_length}, baseline={baseline}")
        except Exception as e:
            print(f"Error loading calibration for sequence {seq}: {e}")
            continue

        # Initialize depth generator
        depth_generator = StereoDepthGenerator(focal_length, baseline)

        # Get list of image files
        left_images = sorted(os.listdir(LEFT_IMG_DIR))
        
        # Process each stereo pair
        for left_img_name in tqdm(left_images, desc=f"Processing stereo pairs for sequence {seq}"):
            # Assume filenames match between left and right (e.g., 0000000000.png)
            right_img_name = left_img_name  # KITTI uses same naming for stereo pairs
            
            left_img_path = os.path.join(LEFT_IMG_DIR, left_img_name)
            right_img_path = os.path.join(RIGHT_IMG_DIR, right_img_name)
            if not os.path.exists(right_img_path):
                print(f"Right image not found: {right_img_path}")
                continue
            
            output_path = os.path.join(OUTPUT_DIR, f"depth_{left_img_name.replace('.png', '.npy')}")
            
            # Process the pair
            depth_generator.process_pair(left_img_path, right_img_path, output_path)

if __name__ == "__main__":
    main()