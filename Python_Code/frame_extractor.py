import cv2
import os
import glob

def variance_of_laplacian(image):
    """Computes the Laplacian of the image and then returns the focus
    measure, which is simply the variance of the Laplacian."""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def extract_frames(video_path, output_dir, sample_rate_sec=1.5, blur_threshold=100.0):
    """
    Extracts frames from a video at a specified interval.
    Filters out frames that are too blurry.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * sample_rate_sec)
    
    frame_count = 0
    saved_count = 0
    
    print(f"Processing {video_name} at {fps:.2f} fps. Sampling every {frame_interval} frames ({sample_rate_sec}s).")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Convert to grayscale for blur detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray)
            
            # Check if image is sharp enough
            if fm > blur_threshold:
                out_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count:04d}.jpg")
                cv2.imwrite(out_path, frame)
                saved_count += 1
            else:
                pass # Skip blurry frame
                
        frame_count += 1

    cap.release()
    print(f"Finished {video_name}. Saved {saved_count} sharp frames.")

if __name__ == "__main__":
    # Adjust path according to where the videos are located
    videos_dir = "Raw_Videos/Dataset"
    output_dir = "sampled_frames"
    
    # Get all video files (assuming mp4, adjust extension if needed)
    video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
    
    # If the videos have different extensions like .avi or .mov, you can add them:
    # video_files.extend(glob.glob(os.path.join(videos_dir, "*.avi")))
    
    if not video_files:
         print(f"No video files found in {videos_dir}")
    else:
        for video_path in video_files:
            # Sample every 1.5 seconds, blur threshold of 100 (adjust threshold based on testing)
            extract_frames(video_path, output_dir, sample_rate_sec=1.5, blur_threshold=100.0)
            
    print("Frame extraction complete. You can now use these frames for annotation.")
