import numpy as np
import ffmpeg
from ffmpeg import Error

class ffmpegProcessor:
    def __init__(self):
        self.cmd = 'ffmpeg-git-20200119-i686-static/ffmpeg'
        
    def extract_audio(self, filename):
        try:
            out, err = (
                ffmpeg
                .input(filename)
                .output('-', format='f32le', acodec='pcm_f32le', ac=1, ar='44100')
                .run(cmd=self.cmd, capture_stdout=True, capture_stderr=True)
            )
        except Error as err:
            print(err.stderr)
            raise
        
        return np.frombuffer(out, np.float32)