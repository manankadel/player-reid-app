import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from src.utils import extract_color_histogram

# Global variable to assign unique IDs to new tracks
next_id = 1

class PlayerTracker:
    def __init__(self, iou_threshold=0.3, max_misses=20, reid_feature_threshold=0.75):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.reid_feature_threshold = reid_feature_threshold
        
        self.active_tracks = []
        self.lost_tracks = []
        
        global next_id
        next_id = 1
        
        print(f"PlayerTracker initialized with Re-ID threshold: {reid_feature_threshold}")

    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0

    def _calculate_feature_similarity(self, hist1, hist2):
        if hist1 is None or hist2 is None:
            return 0.0
        return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)

    def update(self, detections, frame):
        global next_id

        if not self.active_tracks and not detections:
             return []
        if not self.active_tracks:
            for det in detections:
                hist = extract_color_histogram(frame, det[:4])
                self.active_tracks.append({'id': next_id, 'bbox': det[:4], 'misses': 0, 'feature': hist})
                next_id += 1
            return self.get_visible_tracks()

        cost_matrix = np.ones((len(self.active_tracks), len(detections)))
        for i, track in enumerate(self.active_tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1.0 - self._calculate_iou(track['bbox'], det[:4])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_track_indices = set()
        matched_det_indices = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1 - self.iou_threshold):
                track = self.active_tracks[r]
                det_bbox = detections[c][:4]
                track.update({'bbox': det_bbox, 'misses': 0})
                
                new_hist = extract_color_histogram(frame, det_bbox)
                if new_hist is not None and track.get('feature') is not None:
                    track['feature'] = 0.9 * track['feature'] + 0.1 * new_hist
                
                matched_track_indices.add(r)
                matched_det_indices.add(c)

        unmatched_tracks = [self.active_tracks[i] for i in set(range(len(self.active_tracks))) - matched_track_indices]
        for track in unmatched_tracks:
            track['misses'] += 1
            if track['misses'] > self.max_misses:
                self.lost_tracks.append(track)
        
        self.active_tracks = [t for t in self.active_tracks if t['misses'] <= self.max_misses]

        unmatched_dets = [detections[i] for i in set(range(len(detections))) - matched_det_indices]
        
        if unmatched_dets and self.lost_tracks:
            reid_cost_matrix = np.ones((len(unmatched_dets), len(self.lost_tracks)))
            det_hists = [extract_color_histogram(frame, det[:4]) for det in unmatched_dets]

            for i, hist in enumerate(det_hists):
                for j, track in enumerate(self.lost_tracks):
                    similarity = self._calculate_feature_similarity(hist, track['feature'])
                    reid_cost_matrix[i, j] = 1.0 - similarity

            row_ind, col_ind = linear_sum_assignment(reid_cost_matrix)
            
            # --- THE FIX: Defer list modification ---
            reidentified_det_indices = set()
            lost_track_indices_to_remove = set()

            for r, c in zip(row_ind, col_ind):
                if reid_cost_matrix[r, c] < (1.0 - self.reid_feature_threshold):
                    lost_track = self.lost_tracks[c]
                    det = unmatched_dets[r]
                    
                    lost_track.update({'bbox': det[:4], 'misses': 0})
                    self.active_tracks.append(lost_track)
                    
                    reidentified_det_indices.add(r)
                    lost_track_indices_to_remove.add(c)
            
            # Now, modify the lists safely after the loop
            self.lost_tracks = [track for i, track in enumerate(self.lost_tracks) if i not in lost_track_indices_to_remove]
            unmatched_dets = [det for i, det in enumerate(unmatched_dets) if i not in reidentified_det_indices]
            # --- END OF FIX ---

        for det in unmatched_dets:
            hist = extract_color_histogram(frame, det[:4])
            if hist is not None:
                self.active_tracks.append({'id': next_id, 'bbox': det[:4], 'misses': 0, 'feature': hist})
                next_id += 1

        return self.get_visible_tracks()

    def get_visible_tracks(self):
        return [track for track in self.active_tracks if track['misses'] == 0]