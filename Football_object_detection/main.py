from utils import read_video, save_video
from trackers import Tracker


def main():

    video_frames = read_video('Football_object_detection/input_videos/08fd33_4.mp4')


    tracker = Tracker('Football_object_detection/models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                         read_from_stub=True,
                                         stub_path='Football_object_detection/stubs/track_stubs.pkl')

    output_video_track = tracker.draw_annotaations(video_frames, tracks)

    save_video(output_video_track, 'Football_object_detection/output_videos/output_video.avi')

if __name__== '__main__':
    main()
