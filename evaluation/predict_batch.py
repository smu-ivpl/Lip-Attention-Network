import glob
from lipnet.lipreading.videos import Video
from lipnet.lipreading.aligns import Align
from lipnet.lipreading.helpers import text_to_labels
from lipnet.lipreading.visualization import show_video_subtitle
from lipnet.model_lan import LipNet
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizer_v2.adam import Adam
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model_lan import LipNet
import numpy as np
import Averagemeter
from measure import get_cer
from measure import get_wer
import datetime
import os, sys
import random, time, csv

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # Set the GPU 0 to use

random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..','common','predictors','shape_predictor_68_face_landmarks.dat')


PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','common','dictionaries','grid.txt')

lipnet = None
adam = None
spell = None
decoder = None

def predict(weight_path, video):
    global lipnet
    global adam
    global spell
    global decoder

    if lipnet is None:
        lipnet = LipNet(img_c=3, img_w=100, img_h=50, frames_n=75,
                        absolute_max_string_len=32, output_size=28)

        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
        lipnet.model.load_weights(weight_path)

        spell = Spell(path=PREDICT_DICTIONARY)
        decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                          postprocessors=[labels_to_text, spell.sentence])

    X_data       = np.array([video.data]).astype(np.float32) / 255
    input_length = np.array([len(video.data)])

    start_time = time.time()
    y_pred         = lipnet.predict(X_data)
    result         = decoder.decode(y_pred, input_length)[0]
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def predicts(weight_path, videos_path, aligns_path, absolute_max_string_len=32, output_size=28):
    videos = []
    i =0
    videos_100 = glob.glob(os.path.join(videos_path, 's1', '*'))
    videos_100 = videos_100.append(glob.glob(os.path.join(videos_path, 's2', '*')))
    videos_100 = videos_100.append(glob.glob(os.path.join(videos_path, 's20', '*')))
    videos_100 = videos_100.append(glob.glob(os.path.join(videos_path, 's22', '*')))

    for video_path in videos_100:
        print("{} / {} [{}]\nLoading data from disk...".format(i, len(videos_100), video_path))
        videos.append(load(video_path))
        i+=1

    align_hash = {}
    video_ids = []
    for video_path in videos_100:
        video_id = os.path.splitext(video_path)[0].split('/')[-1]
        video_ids.append(video_id)
        align_path = os.path.join(aligns_path, video_id) + ".align"
        align_hash[video_id] = Align(32, text_to_labels).from_file(align_path)

    wer = Averagemeter.AverageMeter()
    cer = Averagemeter.AverageMeter()

    total_time = 0
    total_word = 0


    for idx in range(len(videos_100)):
        print("--------------------------------------------")
        print(f"progress: {idx + 1}/{len(videos_100)}\tvideo path: {videos_100[idx]}\n")

        output, elapsed_time = predict(weight_path, videos[idx])
        total_time += elapsed_time
        total_word += len(output.split())
        groundtruth = align_hash[video_ids[idx]].sentence
        print("hyp: ", output)
        print("ref: ", groundtruth)
        wer.update(get_wer(output, groundtruth), len(groundtruth.split()))
        cer.update(get_cer(output, groundtruth), len(groundtruth))
        print(
            f"cur WER: {wer.val * 100:.1f}\t"
            f"cur CER: {cer.val * 100:.1f}\t"
            f"avg WER: {wer.avg * 100:.1f}\tavg CER: {cer.avg * 100:.1f}\n"
            f"Average inference time per word: {elapsed_time/len(output.split()):.2f}s"
        )


    print("\n\n=======================================================")
    print(f"Average WER for 100 Sentences: {wer.avg * 100:.2f} %")
    print(f"Average CER for 100 Sentences: {cer.avg * 100:.2f} %")
    print(f"Average time required: {total_time/total_word:.2f}s")


def load(video_path):
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)

    print("Data loaded.\n")
    return video

if __name__ == '__main__':
    predicts( "/home/hllee/proj_voice/LipNet/training/unseen_speakers/results-attention/2022:03:23:23:16:36/weights82.h5" ,"/home/hllee/dataset/GRID_corpus/video/", "/home/hllee/dataset/GRID_corpus/align/")
    #predicts(sys.argv[1], sys.argv[2], sys.argv[3])
    # sys.argv[1]: pretrained weights path
    # sys.argv[2]: video path
    # sys.argv[3]: align path