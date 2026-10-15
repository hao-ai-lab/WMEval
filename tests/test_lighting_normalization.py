import shutil
import subprocess
import hashlib
from types import SimpleNamespace

import pytest

from evaluation.lite.normalization import _probe
from evaluation.lite.normalization import normalize_lighting_window, NormalizationError
from evaluation.lite.suite import _normalize_candidate


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
@pytest.mark.parametrize('source_fps', [12, 16, 17, 20, 24, 60])
def test_timestamp_resampling_has_common_fps_without_stretching_time(tmp_path, source_fps):
    source = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                    f'testsrc2=size=64x48:rate={source_fps}:duration=2',
                    '-c:v','libx264','-crf','0',str(source)], check=True, capture_output=True)
    original = source.read_bytes()
    dest = _normalize_candidate(
        case=SimpleNamespace(protocol={'candidate_normalization':'lighting_fps12'}),
        candidate=SimpleNamespace(candidate_video=source, model='test',
                                  case_id='SunRisesAndSets/lighting', seed='42'),
        output_root=tmp_path/'out',
    )
    result = _probe(dest, timeout_sec=30)
    assert result == {'width':64,'height':48,'fps':12.0,'frames':24}
    assert source.read_bytes() == original


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
@pytest.mark.parametrize('source_fps,padding,extra',[(12,0,0),(16,0,1),(17,0,7),(20,0,0),(24,0,13),(60,4,0)])
def test_orbit_window_preserves_speed_and_discards_only_declared_padding(tmp_path,source_fps,padding,extra):
    source=tmp_path/'source.mp4'
    frames=source_fps*10+padding+extra
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',f'testsrc2=size=64x48:rate={source_fps}',
                    '-frames:v',str(frames),'-c:v','libx264','-crf','0',str(source)],check=True,capture_output=True)
    original=source.read_bytes()
    mapping={'schema_version':1,'canonical_zero_frame':padding,'native_fps':source_fps,
             'video_sha256':hashlib.sha256(original).hexdigest()}
    normalized=_normalize_candidate(
        case=SimpleNamespace(protocol={'candidate_normalization':'lighting_orbit_10s_fps12'}),
        candidate=SimpleNamespace(candidate_video=source,model='test',case_id='SunRisesAndSets/lighting_orbit_v2',
                                  seed='42',params={'lighting_time_mapping':mapping}),output_root=tmp_path/'out')
    assert _probe(normalized,timeout_sec=30)=={'width':64,'height':48,'fps':12.,'frames':120}
    assert source.read_bytes()==original
    # A mismatched receipt must fail even when an old normalized file exists.
    mapping['video_sha256']='0'*64
    with pytest.raises(NormalizationError,match='hash'):
        normalize_lighting_window(source,normalized,time_mapping=mapping)


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
def test_short_orbit_is_invalid_instead_of_padded_or_retimed(tmp_path):
    source=tmp_path/'short.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=12:duration=9.5',
                    '-c:v','libx264',str(source)],check=True,capture_output=True)
    mapping={'schema_version':1,'canonical_zero_frame':0,'native_fps':12,
             'video_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    with pytest.raises(NormalizationError,match='short lighting video'):
        normalize_lighting_window(source,tmp_path/'out.mp4',time_mapping=mapping)
    assert not (tmp_path/'out.mp4').exists()
