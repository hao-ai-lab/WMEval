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
        candidate=SimpleNamespace(candidate_video=source,model='test',case_id='Fixture/lighting_orbit',
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


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
@pytest.mark.parametrize('source_fps,extra',[(16,1),(17,7),(24,13)])
def test_orbit_without_receipt_derives_mapping_from_the_video(tmp_path,source_fps,extra):
    source=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',f'testsrc2=size=64x48:rate={source_fps}',
                    '-frames:v',str(source_fps*10+extra),'-c:v','libx264','-crf','0',str(source)],
                   check=True,capture_output=True)
    normalized=_normalize_candidate(
        case=SimpleNamespace(protocol={'candidate_normalization':'lighting_orbit_10s_fps12'}),
        candidate=SimpleNamespace(candidate_video=source,model='test',case_id='Fixture/lighting_orbit',
                                  seed='42',params={}),output_root=tmp_path/'out')
    assert _probe(normalized,timeout_sec=30)=={'width':64,'height':48,'fps':12.,'frames':120}


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
def test_short_orbit_without_receipt_is_still_invalid(tmp_path):
    source=tmp_path/'short.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=16:duration=9.5',
                    '-c:v','libx264',str(source)],check=True,capture_output=True)
    with pytest.raises(NormalizationError,match='short lighting video'):
        _normalize_candidate(
            case=SimpleNamespace(protocol={'candidate_normalization':'lighting_orbit_10s_fps12'}),
            candidate=SimpleNamespace(candidate_video=source,model='test',case_id='Fixture/lighting_orbit',
                                      seed='42',params={}),output_root=tmp_path/'out')


def _luma_ramp(path, *, frames, fps, tag_fps=None, step=1.0):
    """Lossless clip whose frame n has luma round(step * n); tag_fps re-labels the container."""
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',f'nullsrc=s=16x16:r={fps}',
                    '-vf',f"geq=lum='{step}*N':cb=128:cr=128",'-frames:v',str(frames),
                    '-c:v','libx264','-crf','0','-pix_fmt','yuv420p',str(path)],check=True,capture_output=True)
    if tag_fps is not None:
        retagged=path.with_name(path.stem+'_tagged.mp4')
        subprocess.run(['ffmpeg','-v','error','-r',str(tag_fps),'-i',str(path),'-c:v','libx264','-crf','0',
                        '-pix_fmt','yuv420p',str(retagged)],check=True,capture_output=True)
        return retagged
    return path


def _lumas(path):
    import numpy as np
    # Raw Y plane (no range conversion): 16x16 luma + two 8x8 chroma planes per frame.
    raw=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','yuv420p','-'],
                       check=True,capture_output=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(-1,16*16+2*8*8)[:,8*16+8].astype(int)


def _orbit(source,tmp_path,params):
    return _normalize_candidate(
        case=SimpleNamespace(case_id='Fixture/lighting_orbit',
                             protocol={'candidate_normalization':'lighting_orbit_10s_fps12'}),
        candidate=SimpleNamespace(candidate_video=source,model='test',case_id='Fixture/lighting_orbit',
                                  seed='42',params=params),output_root=tmp_path/'out')


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
@pytest.mark.parametrize('rate,frames',[('30000/1001',300),('24000/1001',240)])
def test_rational_frame_rates_cover_ten_seconds_by_timestamp(tmp_path,rate,frames):
    source=tmp_path/'ntsc.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',f'testsrc2=size=64x48:rate={rate}',
                    '-frames:v',str(frames),'-c:v','libx264','-crf','0',str(source)],check=True,capture_output=True)
    assert _probe(_orbit(source,tmp_path,{}),timeout_sec=30)=={'width':64,'height':48,'fps':12.,'frames':120}
    # One native frame short of [0, 10) s is still rejected.
    short=tmp_path/'ntsc_short.mp4'
    subprocess.run(['ffmpeg','-v','error','-i',str(source),'-frames:v',str(frames-1),'-c:v','libx264','-crf','0',
                    str(short)],check=True,capture_output=True)
    with pytest.raises(NormalizationError,match='short lighting video'):
        _orbit(short,tmp_path,{})


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
def test_mislabelled_container_is_retimed_to_the_declared_native_fps(tmp_path):
    # 16 fps generation written into an 8 fps container: 161 frames = 20 s as tagged, 10 s as generated.
    source=_luma_ramp(tmp_path/'ramp.mp4',frames=161,fps=16,tag_fps=8)
    assert _probe(source,timeout_sec=30)['fps']==8.
    lumas=_lumas(_orbit(source,tmp_path,{'candidate_fps_override':16.0}))
    assert len(lumas)==120
    # Output frame k shows native frame round(k * 16 / 12): the whole orbit, at its own speed.
    assert [abs(int(v)-round(k*16/12)) <= 1 for k,v in enumerate(lumas)]==[True]*120
    # Trusting the container instead would score only the first half of the orbit.
    assert lumas[-1] >= 155


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
def test_model_level_zero_frame_skips_warm_up_frames(tmp_path):
    source=_luma_ramp(tmp_path/'warm.mp4',frames=16+160,fps=16)
    lumas=_lumas(_orbit(source,tmp_path,{'lighting_canonical_zero_frame':16}))
    assert len(lumas)==120 and abs(int(lumas[0])-16)<=1 and abs(int(lumas[-1])-(16+round(119*16/12)))<=1
    with pytest.raises(NormalizationError,match='short lighting video'):
        _orbit(source,tmp_path,{'lighting_canonical_zero_frame':17})


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
def test_params_that_contradict_a_receipt_are_rejected(tmp_path):
    from evaluation.lite.suite import SuiteRunError
    source=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=16','-frames:v','161',
                    '-c:v','libx264','-crf','0',str(source)],check=True,capture_output=True)
    receipt={'schema_version':1,'native_fps':16,'canonical_zero_frame':0,
             'video_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    with pytest.raises(SuiteRunError,match='conflicts with the receipt'):
        _orbit(source,tmp_path,{'lighting_time_mapping':receipt,'lighting_canonical_zero_frame':4})
    with pytest.raises(SuiteRunError,match='conflicts with the receipt'):
        _orbit(source,tmp_path,{'lighting_time_mapping':receipt,'candidate_fps_override':10.0})
    with pytest.raises(SuiteRunError,match='nonnegative integer'):
        _orbit(source,tmp_path,{'lighting_canonical_zero_frame':-1})
    # Agreeing params are fine.
    assert _probe(_orbit(source,tmp_path,{'lighting_time_mapping':receipt,'lighting_canonical_zero_frame':0,
                                         'candidate_fps_override':16.0}),timeout_sec=30)['frames']==120
