import cv2
import numpy as np
import imageio
import os

# --- CONFIGURATION ---
VIDEO_PATH = "traffic.mp4"       # Target path for synthetic video generation
GIF_PATH = "tracker_proof.gif"    # Output path for your README proof
MAX_GIF_FRAMES = 40              # Cap frames to keep the GIF size light for GitHub
DISPLAY_SCALE = 0.4               # Scale factor for the real-time preview window

def generate_synthetic_video(file_path, width=640, height=480, fps=20, duration=5):
    """Generates a synthetic traffic video containing high-contrast patterned objects."""
    print("Generating synthetic traffic video asset...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    
    num_frames = fps * duration
    
    # Define moving vehicle parameters (initial positions, speeds, patterns, and colors)
    vehicles = [
        {"x": 30, "y": 140, "vx": 5, "vy": 0, "size": (65, 40), "color": (120, 120, 120)},
        {"x": 500, "y": 280, "vx": -4, "vy": 0, "size": (75, 45), "color": (70, 70, 190)},
        {"x": 200, "y": 300, "vx": -3, "vy": -1, "size": (55, 35), "color": (160, 160, 160)}
    ]
    
    for _ in range(num_frames):
        # Draw dark canvas background with roadway strips
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 120), (width, 220), (45, 45, 45), -1)
        cv2.rectangle(img, (0, 260), (width, 380), (45, 45, 45), -1)
        
        # Add dotted white lane markers to create static landscape features
        for x in range(0, width, 40):
            cv2.line(img, (x, 170), (x + 20, 170), (255, 255, 255), 2)
            cv2.line(img, (x, 320), (x + 20, 320), (255, 255, 255), 2)
            
        # Animate and draw textured vehicles designed to trigger feature descriptors
        for v in vehicles:
            vx_f, vy_f = int(v["x"]), int(v["y"])
            w_v, h_v = v["size"]
            
            # Base vehicle shell
            cv2.rectangle(img, (vx_f, vy_f), (vx_f + w_v, vy_f + h_v), v["color"], -1)
            # High-contrast interior blocks (Windshields & stripes) for structural keypoints
            cv2.rectangle(img, (vx_f + 6, vy_f + 5), (vx_f + 22, vy_f + h_v - 5), (245, 245, 245), -1)
            cv2.rectangle(img, (vx_f + w_v - 22, vy_f + 5), (vx_f + w_v - 6, vy_f + h_v - 5), (25, 25, 25), -1)
            cv2.circle(img, (vx_f + w_v // 2, vy_f + h_v // 2), 6, (0, 255, 255), -1)
            
            # Update dynamics
            v["x"] += v["vx"]
            v["y"] += v["vy"]
            
            # Wrap boundaries around edge axes
            if v["x"] > width: v["x"] = -w_v
            if v["x"] < -w_v: v["x"] = width
            if v["y"] < 0 or v["y"] > height: v["vy"] *= -1

        out.write(img)
        
    out.release()
    print(f"Synthetic video successfully saved to '{file_path}'\n")


def main():
    # Automatically generate the video file if it is missing from the directory
    if not os.path.exists(VIDEO_PATH):
        generate_synthetic_video(VIDEO_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Initialize ORB detector (tuned to 500 features for quick processing loops)
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    prev_gray = None
    prev_kp = None
    prev_des = None
    
    gif_frames = []

    print("Running tracking pipeline... Press 'q' inside the preview window to exit early.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
                # --- REPLACE THIS PORTION INSIDE YOUR WHILE LOOP ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Boost contrast locally using CLAHE so ORB can detect vehicle borders/windows
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # Detect keypoints on the enhanced image
        kp, des = orb.detectAndCompute(gray_enhanced, None)
        display_frame = frame.copy()
        
        if prev_gray is not None and des is not None and prev_des is not None:
            matches = bf.knnMatch(prev_des, des, k=2)
            
            good_matches = []
            active_tracking_count = 0
            
            for m_match in matches:
                if len(m_match) == 2:
                    m, n = m_match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            for match in good_matches:
                pt_prev = tuple(np.round(prev_kp[match.queryIdx].pt).astype(int))
                pt_curr = tuple(np.round(kp[match.trainIdx].pt).astype(int))
                
                # Calculate absolute displacement (movement distance)
                movement_distance = np.sqrt((pt_curr[0] - pt_prev[0])**2 + (pt_curr[1] - pt_prev[1])**2)
                
                # 2. Filter out static background: Only draw if the point actually moved
                if movement_distance > 1.5:  # Adjust threshold based on vehicle speed
                    active_tracking_count += 1
                    # Draw a distinct thick red line showing the trail
                    cv2.line(display_frame, pt_prev, pt_curr, (0, 0, 255), 3)
                    # Draw a bright green dot at the current leading point
                    cv2.circle(display_frame, pt_curr, 5, (0, 255, 0), -1)
                else:
                    # Optional: Draw static background points as faint tiny blue dots
                    cv2.circle(display_frame, pt_curr, 1, (255, 0, 0), -1)
                
            cv2.putText(display_frame, f"Active Motion Points: {active_tracking_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            if kp:
                display_frame = cv2.drawKeypoints(display_frame, kp, None, color=(0, 255, 0))
                
        # Make sure to update the reference to the enhanced gray frame
        prev_gray = gray_enhanced
        prev_kp = kp
        prev_des = des


        # Handle size scaling configurations
        h, w = display_frame.shape[:2]
        resized_frame = cv2.resize(display_frame, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
        cv2.imshow("Classical Vehicle Tracker (ORB + Matcher)", resized_frame)
        
        # Cache frames for compiling the GIF export
        if len(gif_frames) < MAX_GIF_FRAMES:
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            gif_frames.append(rgb_frame)
            
        # Update frame pipeline dependencies
        prev_gray = gray
        prev_kp = kp
        prev_des = des
        
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Render and output the operational validation asset
    if gif_frames:
        print(f"\nCompiling {len(gif_frames)} tracking frames into '{GIF_PATH}'...")
        imageio.mimsave(GIF_PATH, gif_frames, fps=20, loop=0)
        print("GIF export successfully complete! Ready to push to GitHub.")

if __name__ == "__main__":
    main()
