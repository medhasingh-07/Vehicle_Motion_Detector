import cv2
import numpy as np
import os
import imageio


class VehicleMotionTracker:
    def __init__(self, mode='lk_optical_flow', max_corners=200, quality_level=0.3, min_distance=7):
        """
        Modes supported: 'lk_optical_flow' or 'orb_feature_matching'
        """
        self.mode = mode
        
        # Lucas-Kanade Optical Flow Parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Shi-Tomasi Corner Detection Parameters for LK Flow initial features
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )
        
        # ORB Detector & Brute-Force Matcher setup
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
    def track_video(self, input_video_path, output_gif_path, max_frames=150, fps=15):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Error opening video file: {input_video_path}")

        # Read initial frame
        ret, old_frame = cap.read()
        if not ret:
            print("Failed to read video stream.")
            return

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        # Create a mask image for drawing optical flow tracks
        mask = np.zeros_like(old_frame)
        
        frames_for_gif = []
        frame_count = 0
        
        if self.mode == 'lk_optical_flow':
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)
        elif self.mode == 'orb_feature_matching':
            kp_prev, des_prev = self.orb.detectAndCompute(old_gray, None)

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output_frame = frame.copy()

            if self.mode == 'lk_optical_flow':
                if p0 is not None and len(p0) > 0:
                    # Compute Optical Flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)
                    
                    if p1 is not None:
                        # Select good points
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]
                        
                        # Draw vector tracks
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                            output_frame = cv2.circle(output_frame, (int(a), int(b)), 4, (0, 0, 255), -1)
                        
                        # Update previous points and frame
                        p0 = good_new.reshape(-1, 1, 2)
                    
                    # Refresh tracking keypoints if too few remain
                    if len(p0) < 20:
                        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
                        mask = np.zeros_like(frame)

            elif self.mode == 'orb_feature_matching':
                kp_curr, des_curr = self.orb.detectAndCompute(frame_gray, None)
                
                if des_prev is not None and des_curr is not None and len(des_prev) > 1 and len(des_curr) > 1:
                    # K-Nearest Neighbor matching
                    matches = self.bf_matcher.knnMatch(des_prev, des_curr, k=2)
                    
                    # Apply Lowe's Ratio Test to reject ambiguous matches
                    good_matches = []
                    for m_n in matches:
                        if len(m_n) == 2:
                            m, n = m_n
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)
                    
                    # Plot trajectory vectors on frame
                    for match in good_matches:
                        pt1 = tuple(np.round(kp_prev[match.queryIdx].pt).astype(int))
                        pt2 = tuple(np.round(kp_curr[match.trainIdx].pt).astype(int))
                        cv2.line(output_frame, pt1, pt2, (255, 0, 0), 2)
                        cv2.circle(output_frame, pt2, 3, (0, 255, 255), -1)

                kp_prev, des_prev = kp_curr, des_curr

            # Overlay historical motion tracks onto current frame
            combined_vis = cv2.add(output_frame, mask)
            
            # Convert BGR to RGB for imageio (GIF creation)
            rgb_vis = cv2.cvtColor(combined_vis, cv2.COLOR_BGR2RGB)
            frames_for_gif.append(rgb_vis)
            
            old_gray = frame_gray.copy()
            frame_count += 1

        cap.release()

        # Save output as animated GIF proof
        print(f"Exporting proof demo to: {output_gif_path} ...")
        imageio.mimwrite(output_gif_path, frames_for_gif, fps=fps)
        print("GIF Export Complete!")


if __name__ == "__main__":
    # Example usage:
    INPUT_VIDEO = "vehicles_sample.mp4"
    OUTPUT_GIF = "assets/vehicle_tracking_demo.gif"
    
    os.makedirs("assets", exist_ok=True)

    # Initialize and run vehicle tracker using LK Optical Flow
    tracker = VehicleMotionTracker(mode='lk_optical_flow')
    tracker.track_video(
        input_video_path=INPUT_VIDEO,
        output_gif_path=OUTPUT_GIF,
        max_frames=120,
        fps=12
    )