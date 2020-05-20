import numpy as np
import cv2
import av


def frame_to_arr(frame, dim, resize, normalize, interpolation):
    h, w, _ = dim
    img = frame.to_image().convert('RGB')
    img = np.array(img, np.uint8)
    if resize:
        img = cv2.resize(img, (h, w), interpolation=interpolation)
    if normalize:
        img = img.astype(np.float32)
        img /= 128.
        img -= 1.
    img = img.reshape((1, *dim))
    return img

def video_to_arrays(path, dim=(128, 128, 3), only_key_frames=True, frames_count=3, interpolation=cv2.INTER_CUBIC, resize=True, normalize=True):
    with av.open(path) as container:
        arrs = np.zeros((0, *dim), np.float32)
        stream = container.streams.video[0]  # 0 - видео, 1,2 - аудио и субтитры
        stream.codec_context.skip_frame = 'NONKEY' if only_key_frames else 'DEFAULT'
        # В stream.frames последний кадр всегда пустой
        frames_count = stream.frames - 1 if frames_count > stream.frames else frames_count
        rand_frames = np.arange(stream.frames) if only_key_frames else np.random.choice(stream.frames - 1, frames_count, replace=False)
        for frame in container.decode(stream):
            if frame.index in rand_frames:
                arr = frame_to_arr(frame, dim, resize, normalize, interpolation)
                arrs = np.concatenate((arrs, arr), axis=0)
                if arrs.shape[0] == frames_count: break
        return arrs
